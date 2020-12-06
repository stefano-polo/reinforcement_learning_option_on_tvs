from gym.envs.registration import register

#LIST OF ALL MY ENVIROMENTS

# FE agents
# ----------------------------------------
register(
    id='CompoundOption-v0',                    #name of the enviroment
    entry_point='envs.compound_option:CompoundOption',         #location of the file
)

register(
    id='VanillaOption-v0',
    entry_point='envs.call_option:PlainVanillaOption',
)

register(
    id='TVS_simple-v0',
    entry_point='envs.tvs_simple_env:TVS_simple',
)

register(
    id='TVS-v0',
    entry_point='envs.tvs_env:TVS_environment',
)

register(
    id='TVS_2assets-v0',
    entry_point='envs.tvs_2market_env:TVS_environment2',
)

register(
    id='TVS_3assets-v0',
    entry_point='envs.tvs_3market_env:TVS_enviroment3',
)
register(
    id='TVS_lv-v0',
    entry_point='envs.tvs_lv_env:TVS_LV',
)