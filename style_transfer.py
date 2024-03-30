from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# %% [markdown]
# For loading the Stable Diffusion using Diffusers, follow the instuctions https://huggingface.co/blog/stable_diffusion and update MY_TOKEN with your token.

# %%
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
remote = "CompVis/stable-diffusion-v1-4"
local = "./stable-diffusion-v1-4"
ldm_stable = StableDiffusionPipeline.from_pretrained(local, scheduler=scheduler).to(device)
tokenizer = ldm_stable.tokenizer

# %% [markdown]

# %%

class LocalBlend:
    
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th


        
        
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_word)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))

# %% [markdown]
# ## Null Text Inversion code

# %%
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    print(h,w,'h and w')
    # left = min(left, w-1)
    # right = min(right, w - left - 1)
    # top = min(top, h - left - 1)
    # bottom = min(bottom, h - top - 1)
    # image = image[top:h-bottom, left:w-right]
    # h, w, c = image.shape
    # print(h,w,'h and w')

    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        
    
    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None

null_inversion = NullInversion(ldm_stable)


# %% [markdown]
# ## Infernce Code

# %%
@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent



def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True, description='',folder=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator,description=description)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings)
    if verbose:
        ptp_utils.view_images(images,description=description,folder=folder)
    return images, x_t

# %%
seed = 1024
g_cpu = torch.Generator().manual_seed(seed)
import argparse

self_ratio = 0.5
torch.cuda.empty_cache()
image_path = "./example_images/cats.png"
prompt = "A photo of two kittens are playfully sparring on a stump, with one leaping towards the other."

# null-text inversion,将输入的图片和prompt进行inversion，得到图片对应的噪声和可以引导这个噪声重建图片的无条件embeding
(image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,200,0), verbose=True)

print("Modify or remove offsets according to your image!")

# %%
# 将null-text inversion得到的噪声和无条件embeding输入stable-diffusion中，得到重建图片的结果。
# 从左到右依次是ground-truth，vqae的重建结果，和我们使用的null-text inversion的重建结果
prompts = [prompt,prompt]
controller = AttentionStore()
image_inv, x_t = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, verbose=False,generator=g_cpu) 
print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image")
ptp_utils.view_images([image_gt, image_enc, image_inv[0]],description='view')
ptp_utils.view_images(image_gt)
# show_cross_attention(controller, 16, ["up", "down"])

# 接下来是使用修改后的prompt以及inversion得到的噪声和无条件embeding对输入的图片进行修改

# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["A photo of two kittens are playfully sparring on a stump, with one leaping towards the other.",
        "A photo of two kittens are playfully sparring on a stump, with one leaping towards the other, in Van-Gogh drawing style."
       ]

cross_replace_steps = {'default_': 0.5}
self_replace_steps = .3

blend_word = None
# blend_word = ((('stormy',), ("ice",))) # for local edit
eq_params = {"words": ("Van-Gogh", "drawing"), "values": (5, 5)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])

# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["A majestic white horse with a flowing mane trotting across a green field.",
        "A majestic white horse, in the Renaissance oil painting style."
       ]

cross_replace_steps = {'default_': 0.2}
self_replace_steps = .6

blend_word = None
# blend_word = ((('stormy',), ("ice",))) # for local edit
eq_params = {"words": ("Renaissance", "oil", "painting"), "values": (5, 4, 3)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["Portrait of a woman with emphasis on her expressive eyes, subtle lips, and elegant figure.",
        "Portrait of a woman with cartoon-style glasses and striking blue eyes, subtle lips, and elegant figure."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('stormy',), ("ice",))) # for local edit
eq_params = {"words": ("cartoon-style", "glasses", "blue"), "values": (5, 1, 0.3)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])

# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["A portrait of a woman with evocative eyes, subtle lips and an elegant figure on a black background.",
        "A portrait of a woman with cyberpunk authentic eyes, rose-colored lips and an elegant figure on a dreamy background."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('eyes',), ("eyes",))) # for local edit
eq_params = {"words": ("cyberpunk", "rose-colored"), "values": (1, 2)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])


# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["A portrait of a woman with evocative eyes, subtle lips and an elegant figure on a black background.",
        "A portrait of a old woman with evocative eyes, subtle lips and an elegant figure on a white background."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('eyes',), ("Van Gogh",))) # for local edit
# eq_params = {"words": ("geometric", "abstraction"), "values": (1, 2)}  
eq_params = {"words": ("white", ), "values": (5, )} 
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])

# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["Intense young man with a pensive stare, wearing a dark jacket.",
        "Intense young man with a pensive stare, wearing a dark jacket, in the ink painting style."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('stormy',), ("ice",))) # for local edit
eq_params = {"words": ("ink","painting"), "values": (1.5, )}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["Close-up of a brown and white cow with gentle eyes in a field.",
        "Cybernetic cow with futuristic elements, in the anime style."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('stormy',), ("ice",))) # for local edit
# eq_params = {"words": ("fairy", "tale"), "values": (5, 5)}  
eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])

# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["Enchanted garden gazebo surrounded by vibrant flowers and trees.",
        "Dilapidated garden gazebo surrounded by monochrome botanicals, in chiaroscuro style."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('autumn',), ("springtime",))) # for local edit
eq_params = {"words": ("Dilapidated", ), "values": (3.75, )}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])

# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["Enchanted garden gazebo surrounded by vibrant flowers and trees.",
        "Snow-covered garden gazebo surrounded by flowers and trees in a wintry scene."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('autumn',), ("springtime",))) # for local edit
# eq_params = {"words": ("cartoon", ), "values": (1.5, )}  
eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["Naval battle with sailing ships and cannon fire, in stormy weather.",
        "Naval battle with sailing ships and cannon fire,  in the ice and snow weather."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

# blend_word = None
blend_word = ((('stormy',), ("ice",))) # for local edit
eq_params = {"words": ("ice", "snow"), "values": (15, 5)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])

# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["Naval battle with sailing ships and cannon fire, in stormy weather.",
        "Naval battle with sailing ships and cannon fire, in the cartoon style."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('autumn',), ("springtime",))) # for local edit
eq_params = {"words": ("cartoon", ), "values": (1.5, )}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["Sunset over a misty autumn forest with a pond.",
        "Iceberg over a misty autumn forest with a pond."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('autumn',), ("springtime",))) # for local edit
# eq_params = {"words": ("Iceberg", ), "values": (10, )}  
eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["Sunset over a misty autumn forest with a pond.",
        "Sunset over a misty autumn forest with a pond, in cyberpunk style "
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('Iceberg',), ("Sunset",))) # for local edit
eq_params = {"words": ("cyberpunk", ), "values": (4, )}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])

# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["a photo of a young Asian man with short hair and white t-shirt.",
        "a photo of a young Asian man with long hair and black t-shirt, in the cyberpunk style"
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

# blend_word = None
blend_word = ((('white',), ("black",))) # for local edit
eq_params = {"words": ("black", "cyberpunk"), "values": (1, 10)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["a photo of a young Asian man with short hair and white t-shirt.",
        "a photo of a young Asian man with short hair and white t-shirt, in the Impressionist style."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('building',), ("building",))) # for local edit
eq_params = {"words": ("Impressionist", ), "values": (10, 10)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])

# %%
# 全局编辑，风格迁移，梵高风格
prompts = ["a photo of a young Asian man with short hair and white t-shirt.",
        "a photo of a young Asian man with short hair and white t-shirt, in the pop art style."
       ]

cross_replace_steps = {'default_': 0.85}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('building',), ("building",))) # for local edit
eq_params = {"words": ("pop art",), "values": (17.5,)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])

# %%
# 全局编辑，风格迁移.将主楼后面的背景换成银河系
prompts = ["A white horse in the background.",
        "A white horse in the primeval forest."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('building',), ("building",))) # for local edit
eq_params = {"words": ("forest",), "values": (20,)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])

# %%
# 局部编辑，使用原prompt和新prompt的对应位置的token构造一个mask，将北理工的主楼改成一个城堡
prompts = ["A white horse in the background.",
        "A rainbow horse in the background."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('building',), ("building",))) # for local edit
eq_params = {"words": ("rainbow"), "values": (2,3)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])

# %%
# 局部编辑，使用原prompt和新prompt的对应位置的token构造一个mask，将北理工的主楼改成一个城堡
prompts = ["A white horse in the background.",
        "A deer in the ocean background."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = 0.4

# blend_word = None
blend_word = ((('horse',), ("deer",))) # for local edit
eq_params = {"words": ("deer", ), "values": (3,)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 局部编辑，使用原prompt和新prompt的对应位置的token构造一个mask，将北理工的主楼改成一个城堡
prompts = ["A photo of a sandstone temple with detailed carvings under a clear blue sky.",
        "A photo of a sandstone temple with detailed carvings under a clear blue sky, in an colorful anime style."
       ]

cross_replace_steps = {'default_': 0.9}
self_replace_steps = 0.1

# blend_word = None
blend_word = ((('temple',), ("temple",))) # for local edit
eq_params = {"words": ("anime",), "values": (10,)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])

# %%
# 局部编辑，使用原prompt和新prompt的对应位置的token构造一个mask，将北理工的主楼改成一个城堡
prompts = ["A white horse in the background.",
        "A white horse in the ocean background."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('background',), ("ocean",))) # for local edit
eq_params = {"words": ("ocean",), "values": (30,)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 局部编辑，将北理工主楼的风格改成中国风
prompts = ["A colossal black statue of Shiva under a blue sky with white clouds.",
        "A colossal black statue of Shiva under the blue sky with white clouds, in the watercolor style."
       ]

cross_replace_steps = {'default_': .3}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('blue',), ("red",))) # for local edit
eq_params = {"words": ("watercolor", ), "values": (30, )}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 控制prompt中某个单词对生成图片的影响程度。修改原prompt中的cloudy对应的attention map注入的权重，从而控制新的图片中云的数量。
# 从左到右cloudy的权重分别是2,5,10,20
prompts = ["A white university building with a red signboard in front that says Beijing Institute of Technology under a cloudy sky.",
        "A white university building with a red signboard in front that says Beijing Institute of Technology under a cloudy sky"
       ]
lb = LocalBlend(prompts, ("sky", "sky"))
attention_weight = [2, 5, 10, 20]
display_imgs = np.zeros((1 + len(attention_weight), 512, 512, 3))
for i, att in enumerate(attention_weight):
    equalizer = get_equalizer(prompts[1], ("cloudy",), (att,))
    controller = AttentionReweight(prompts, 50, cross_replace_steps=.8,
                                self_replace_steps=.4,
                                equalizer=equalizer)
    images,_ = run_and_display(prompts, controller, latent=x_t, uncond_embeddings=uncond_embeddings, generator=g_cpu,verbose=False, run_baseline=False)
    if i == 0: display_imgs[0:2,:] = images
    else: display_imgs[i + 1] = images[-1]
ptp_utils.view_images(display_imgs)
# %%
# %%
prompts = ["A man wearing a black shirt and suspenders with a spinning basketball on his fingers",
        "A man wearing a black shirt and suspenders with a bright flame on his fingers"
       ]

cross_replace_steps = {'default_': .4}
self_replace_steps = self_ratio

# blend_word = None
blend_word = ((('basketball',), ("flame",))) # for local edit
eq_params = {"words": ("bright", "flame"), "values": (20,20)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])



# %%
prompts = ["A man wearing a black shirt and suspenders with a spinning basketball on his fingers",
        "A long hair woman wearing a black shirt and suspenders with a spinning basketball on his fingers"
       ]

cross_replace_steps = {'default_': .4}
self_replace_steps = self_ratio

# blend_word = None
blend_word = ((('man',), ("woman",))) # for local edit
eq_params = {"words": ("woman", "long", "hair"), "values": (2,20,2)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])

# %%

prompts = ["A boy with blue jacket, black short hair ,a black bag and excited expression",
            "A boy with blue jacket, black long bangs,a black bag and excited expression"
        ]

cross_replace_steps = {'default_': .2, }

blend_word = ((('hair',), ("bangs",)))
eq_params = {"words": ("long", "bangs"), "values": (20, 100)}  # amplify attention to the words "silver" and "sculpture" by *2 
 
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
# %%
prompts = ["A boy with blue jacket, black short hair ,a black bag and excited expression",
            "A boy with blue jacket,black glasses,black short hair,a black bag and excited expression"
        ]

cross_replace_steps = {'default_': .8, }

blend_word = None
eq_params = {"words": ("glasses"), "values": (10,)}  # amplify attention to the words "silver" and "sculpture" by *2 
 
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
# %%
prompts = ["A boy with blue jacket, black short hair ,a black bag and excited expression",
            "A boy with blue jacket,black short hair,a black bag,a white baseball hat and excited expression"
        ]

cross_replace_steps = {'default_': .8, }

# blend_word = ((('hair',), ("hair",)))
blend_word = None
eq_params = {"words": ("white","baseball", "hat"), "values": (25,5,5)}  # amplify attention to the words "silver" and "sculpture" by *2 
 
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
# %%
# 小女孩水墨画
prompts = ["A cheerful little girl with blue eyes and curly hair smiling and pulling at her cheeks.",
        "A cheerful little girl with blue eyes and curly hair smiling and pulling at her cheeks, in watercolor painting style."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('building',), ("building",))) # for local edit
eq_params = {"words": ("watercolor","painting"), "values": (5,0.2)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 小女孩
prompts = ["A cheerful little girl with blue eyes and curly hair smiling and pulling at her cheeks.",
        "A cheerful little girl with blue eyes and curly hair smiling and pulling at her cheeks, in pop art style ."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('building',), ("building",))) # for local edit
eq_params = {"words": ("pop","art"), "values": (5,0.2)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 小女孩波普艺术风格
prompts = ["A cheerful little girl with blue eyes and curly hair smiling and pulling at her cheeks.",
        "A cheerful little girl with blue eyes and curly hair smiling and pulling at her cheeks, in pop art style ."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('building',), ("building",))) # for local edit
eq_params = {"words": ("pop","art"), "values": (5,5)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 小女孩 淡淡的微光和马赛克般的背景
prompts = ["A cheerful little girl with blue eyes and curly hair smiling and pulling at her cheeks.",
        "A cheerful little girl with blue eyes and curly hair smiling and pulling at her cheeks, with a touch of golden shimmer and a mosaic background."
       ]

cross_replace_steps = {'default_': 0.35}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('building',), ("building",))) # for local edit
eq_params = {"words": ("golden", "shimmer","mosaic",), "values": (10,10,3)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 小女孩 梦境怪诞
prompts = ["A cheerful little girl with blue eyes and curly hair smiling and pulling at her cheeks.",
        "A cheerful little girl blending elements of Salvador Dali's dreamlike landscapes, ultra high resolution"
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('building',), ("building",))) # for local edit
eq_params = {"words": ("Salvador Dali's","dreamlike" ,), "values": (10,10)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])

# %%
# 小女孩 文艺复兴时期
prompts = ["A cheerful little girl with curly hair smiling and pulling at her cheeks.",
        "A cheerful little girl with curly hair smiling and pulling at her cheeks, in Renaissance oil paintings style."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('building',), ("building",))) # for local edit
eq_params = {"words": ("Renaissance","oil","paintings"), "values": (5,3,3)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
# 小女孩 女人
prompts = ["A cheerful little girl with curly hair smiling and pulling at her cheeks.",
        "A old woman with curly hair smiling and pulling at her cheeks."
       ]

cross_replace_steps = {'default_': 0.8}
self_replace_steps = self_ratio

blend_word = None
# blend_word = ((('building',), ("building",))) # for local edit
eq_params = {"words": ("old","woman"), "values": (3,3)}  
# eq_params = None
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,folder=image_path.split('/')[-1])
ptp_utils.view_images(images[1])
# %%
prompts = ["Indoors, a man wearing pink clothes, black glasses and champagne-colored headphones",
           "Indoors, a long hair man wearing pink clothes, black glasses and champagne-colored headphones, in cartoon style, extra high quality and pleasing"
        ]

cross_replace_steps = {'default_': .8, }

blend_word = None
eq_params = {"words": ("cartoon", 'style','long','hair' ), "values": (2,2,2,2)}  # amplify attention to the words "silver" and "sculpture" by *2 
 
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,description='long hair',folder=image_path.split('/')[-1])

prompts = ["Indoors, a man wearing pink clothes, black glasses and champagne-colored headphones",
           "Indoors, a smiling man wearing pink clothes, black glasses and champagne-colored headphones, in cartoon style, extra high quality and pleasing"
        ]

cross_replace_steps = {'default_': .8, }

blend_word = None
eq_params = {"words": ("cartoon", 'style','smiling' ), "values": (2,2,2)}  # amplify attention to the words "silver" and "sculpture" by *2 
 
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,description='smiling',folder=image_path.split('/')[-1])

prompts = ["Indoors, a man wearing pink clothes, black glasses and champagne-colored headphones",
           "Indoors, a man wearing a joker mask, pink clothes, black glasses and champagne-colored headphones, in cartoon style, extra high quality and pleasing"
        ]

cross_replace_steps = {'default_': .8, }

blend_word = None
eq_params = {"words": ("cartoon", 'style','joker','mask' ), "values": (2,2,2,2)}  # amplify attention to the words "silver" and "sculpture" by *2 
 
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,description='joker mask',folder=image_path.split('/')[-1])

prompts = ["Indoors, a man wearing pink clothes, black glasses and champagne-colored headphones",
           "Forest, a man wearing pink clothes, black glasses and champagne-colored headphones, in cartoon style, extra high quality and pleasing"
        ]

cross_replace_steps = {'default_': .8, }

blend_word = None
eq_params = {"words": ("cartoon", 'style','Forest' ), "values": (2,2,2)}  # amplify attention to the words "silver" and "sculpture" by *2 
 
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,generator=g_cpu,description='forest',folder=image_path.split('/')[-1])