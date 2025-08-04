from gymnasium.envs.registration import register

def registration_envs() -> None:
    register(
        id='2DBpp-v1',                                  
        entry_point='envs.bpp:BppEnv',   
        kwargs={
            'bin_w': 10,
            'bin_h': 10,
        }
    )
    register(
        id='2DBpp-v2',                                  
        entry_point='envs.bpp:BppEnv',   
        kwargs={
            'bin_w': 20,
            'bin_h': 20,
        }
    )
    register(
        id='2DBpp-v3',                                  
        entry_point='envs.bpp:BppEnv',   
        kwargs={
            'bin_w': 40,
            'bin_h': 40,
        }
    )
    register(
        id='2DBpp-v4',                                  
        entry_point='envs.bpp:BppEnv',   
        kwargs={
            'bin_w': 32,
            'bin_h': 50,
        }
    )
