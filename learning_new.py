heroes = ['Hitler', 'God', 'Jesus', 'Myself']
real = ['Jewish', 'Never existed', 'What?', 'Best']

for i, index in zip(enumerate(heroes, 1), enumerate(real, 1)):
    print(i, index)
