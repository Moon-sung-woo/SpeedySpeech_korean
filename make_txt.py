f = open('transcript.v.1.4.txt', 'r', encoding='utf-8')
wf = open('code/datasets/data/kss/kss_script.v.1.4.txt', 'w', encoding='utf-8')

lines = f.readlines()
for line in lines:
    line = line[2:]
    re_line = line.split('|')[:2]
    print('|'.join(re_line))
    wf.write('|'.join(re_line) + '\n')

wf.close()
f.close()