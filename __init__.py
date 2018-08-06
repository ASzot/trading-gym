from gym.envs.registration import register

register(
    id='SeriesEnv-v0',
    entry_point='custom_gym_envs.series_env:SeriesEnv',
)

