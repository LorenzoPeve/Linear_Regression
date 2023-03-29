# cd /home/lpeve/anaconda3/envs/pymc_env/bin/acti
# /home/lpeve/anaconda3/envs/pymc_env//bin/jupyter-notebook
# conda activate pymc_env

def remove_dups(l1,l2):

    for e in l1:
        if e in l2:
            l1.remove(e)

l1 = [1,2,3,4]
l2 = [1,2,5,6]
remove_dups(l1,l2)