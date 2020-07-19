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
    entry_point='envs.plain_vanilla:PlainVanillaOption',
)

register(
    id='VanillaHedge-v0',
    entry_point='envs:Vanilla',
)
