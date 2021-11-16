import numpy as np
from numpy.lib.financial import _rbl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
from matplotlib.pyplot import figure

matplotlib.rcParams.update({'font.size': 22})
figure(figsize=(14, 12), dpi=80)

def print_usage():
  print('Usage: adv_plotter.py <path_to_data>')

def rearrange(in_list):
  # made order: CW DF FGSM Elastic BasicIt, Universal
  # Order to do: BasicIt, FGSM, DF, Elastic, Universal, CW
  in_list[0], in_list[1], in_list[2], in_list[3], in_list[4], in_list[5] = in_list[4], in_list[2], in_list[1], in_list[3], in_list[5], in_list[0]

def main(argv):

  if(len(sys.argv) < 2):
    print_usage()
    exit()

  data = str(sys.argv[1])
  df = pd.read_csv(data)

  attacks = df['Attack'].unique()
  models = df['Model'].unique()

  # print(attacks)
  # print(models)

  att_list = attacks
  rearrange(att_list)

  for model in models:
    df2 = df.loc[df['Model'] == model]
    fb_on_fb = df2['FB-on-FB'].to_list()
    fb_on_nncf = df2['FB-on-NNCF'].to_list()
    nncf_on_nncf = df2['NNCF-on-NNCF'].to_list()
    nncf_on_fb = df2['NNCF-on-FB'].to_list()
    rearrange(fb_on_fb)
    rearrange(fb_on_nncf)
    rearrange(nncf_on_nncf)
    rearrange(nncf_on_fb)

    fig = plt.figure()
    plt.figure(figsize=(15, 13))
    plt.plot(att_list, fb_on_fb, '-o', label='with-FB-on-FB')
    plt.plot(att_list, fb_on_nncf, '-^', label='with-FB-on-NNCF')
    plt.plot(att_list, nncf_on_nncf, '-<', label='with-NNCF-on-NNCF')
    plt.plot(att_list, nncf_on_fb, '->', label='with-NNCF-on-FB')
    plt.xticks(rotation=90)
    plt.xlabel("Attack")
    plt.ylabel("Accuracy")
    plt.title(model)
    plt.xticks(rotation=70)
    plt.legend()
    plt.tight_layout()
    plt.savefig(model + '.png')
    fb_on_fb = []
    fb_on_nncf = []
    nncf_on_nncf = []
    nncf_on_fb = []
    print(model + ' done.')

if __name__ == "__main__":
   main(sys.argv[1:])