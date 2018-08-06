from series_env import SeriesEnv



series_env = SeriesEnv()

series_env.rand_seed()
cur_price = series_env.reset()[0]
cur_money = 1000.0

print('Market simulator')
print('Actions: 1-buy/sell, 0-do nothing, q-quit')
print('-' * 20)
print('')

while True:
    cur_date = series_env.get_cur_date().strftime('%Y-%m-%d %H:%M:%S')
    print('')
    print('Price diff', cur_price)

    holding_str = ''
    if series_env.is_holding:
        holding_str = '(holding)'
    print('%s: $%.2f %s' % (cur_date, cur_money, holding_str))
    action = input('Action: ')
    if action == 'q':
        break

    if 'n' in action:
        skip_steps = int(action[:-1])
        for i in range(skip_steps):
            series_env.step(0)
        continue

    action = int(action)
    obs, reward, done, _ = series_env.step(action)
    if done:
        print('Closed out trade with reward of %.2f' % reward)
        print('')

    cur_money += reward

    cur_price = obs[0]



