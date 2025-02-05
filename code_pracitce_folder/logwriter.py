import json

my_dict = {}
for i in range(10):
    my_dict[f'epoch_{i}'] = {i+1 : i}

for i in range(10):
    my_dict[f'epoch_{i}'].update({i*100 : i})

savedir = './log'
with open(f'{savedir}/my_dict.json', 'w') as f:
    json.dump(my_dict, f)
