# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("gpt2-small")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab

data = load_dataset("stas/openwebtext-10k")

tokenized_data = utils.tokenize_and_concatenate(data["train"], model.tokenizer, max_length=256)
tokenized_data = tokenized_data.shuffle(42)
# %%
layer = 8 # 7
head = 6 # 9
tokens = tokenized_data[:48]["tokens"]
logits, cache = model.run_with_cache(tokens)
# %%
max_pattern, argmax_pattern = cache.stack_activation("pattern")[layer, :, head, :, 1:].max(-1)
histogram(max_pattern.flatten())
scatter(x=max_pattern.flatten(), y=einops.repeat(torch.arange(len(tokens[0])), "len -> (batch len)", batch=len(tokens)))
# %%
s = "When John and Mary went to the store, John gave a bottle of milk to Mary. When Alice and Daniel walked to the park, Daniel threw the ball to Alice."
for layer, head in [(8, 10), (8, 6), (7, 9), (7, 3)]:
    temp_logits, temp_cache = model.run_with_cache(s)
    imshow(temp_cache["pattern", layer][0, head], title=f"Layer {layer}, Head {head}", x=nutils.process_tokens_index(s), y=nutils.process_tokens_index(s))
# %%
ave_head_z = cache.stack_activation("z")[layer, :, :, head].mean([0, 1])
def mean_ablate_head(z, hook):
    z[:, :, head, :] = ave_head_z
    return z
logits, cache = model.run_with_cache(tokens)
model.add_hook(utils.get_act_name("z", layer), mean_ablate_head)
abl_logits, abl_cache = model.run_with_cache(tokens)
model.reset_hooks()
# %%

normal_clps = model.loss_fn(logits, tokens, True)
abl_clps = model.loss_fn(abl_logits, tokens, True)

token_df = nutils.make_token_df(tokens, 6, 3, model)
token_df = token_df.query(f"pos <= {token_df.pos.max() - 1}")
token_df["normal_clp"] = to_numpy(normal_clps.flatten())
token_df["abl_clp"] = to_numpy(abl_clps.flatten())
token_df["loss_diff"] = token_df["abl_clp"] - token_df["normal_clp"]
token_df["max_pattern"] = to_numpy(max_pattern[:, :-1].flatten())
token_df["argmax_index_offset"] = (np.arange(len(tokens[0])-1)[None, :] - to_numpy(argmax_pattern[:, :-1]) - 1).flatten()
x = tokens[np.arange(len(tokens))[:, None], argmax_pattern[:, :-1].cpu()+1]
token_df["argmax_token"] = nutils.list_flatten([model.to_str_tokens(x[i]) for i in range(len(x))])
nutils.show_df(token_df.sort_values("loss_diff", ascending=False).head(20))
# %%
top_k = 100
top_df = token_df.sort_values("loss_diff", ascending=False).head(100)
batches = top_df.batch.values
poses = top_df.pos.values
loss_diff = top_df.loss_diff.values
line(loss_diff)

normal_resid_stack, resid_labels = cache.decompose_resid(apply_ln=True, return_labels=True)
abl_resid_stack, resid_labels = abl_cache.decompose_resid(apply_ln=True, return_labels=True)
diff_resid_stack = normal_resid_stack - abl_resid_stack
top_resid_stack_diff = diff_resid_stack[:, batches, poses, :]
print(top_resid_stack_diff.shape)

next_token = tokens[batches, poses+1]
W_U_next = model.W_U[:, next_token]
print(W_U_next.shape)
line((top_resid_stack_diff * W_U_next.T).sum(-1).mean(-1), x=resid_labels)
# %%
x = (top_resid_stack_diff * W_U_next.T).sum(-1)
histogram(x[-8:].T, marginal="box", barmode="overlay")
# %%
top_normal_z = cache["z", 9][batches, poses, :, :]
top_abl_z = abl_cache["z", 9][batches, poses, :, :]
top_diff_z = top_normal_z - top_abl_z
top_head_dla = einops.einsum(top_diff_z, model.W_O[9], W_U_next, "batch head d_head, head d_head d_model, d_model batch -> head batch")
line(top_head_dla)
# %%
