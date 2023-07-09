from gym.envs.registration import register

# LIST OF ALL MY ENVIROMENTS

# FE agents
# ----------------------------------------
register(
    id="VanillaOption-v0",  # name of the environment
    entry_point="envs.plain_vanilla_env:PlainVanillaOption",  # location of the file
)

register(
    id="TVS_BS-v0",
    entry_point="envs.tvs_bs_env:TVS_BS_ENV",
)

register(
    id="TVS_LV-v0",
    entry_point="envs.tvs_lv_env:TVS_LV_ENV_reward1",
)

register(
    id="TVS_LV-v2",
    entry_point="envs.tvs_lv_env_new_reward:TVS_LV_ENV_reward2",
)
