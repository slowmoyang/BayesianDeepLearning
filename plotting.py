import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from uncertainty import compute_variation_ratio
from uncertainty import compute_predictive_entropy
from uncertainty import compute_mutual_information


def draw_uncertainty(is_true_pos, is_false_neg,
                     is_true_neg, is_false_pos,
                     uncertainty,
                     uncertainty_name,
                     tag, title):
    quark = [
        uncertainty[is_true_pos],
        uncertainty[is_false_neg]
    ]

    gluon = [
        uncertainty[is_true_neg],
        uncertainty[is_false_pos]
    ]

    fig, axarr = plt.subplots(ncols=2, figsize=(16, 6))
    suptitle = fig.suptitle(title, fontsize="x-large")

    quark_hists, _, _ = axarr[0].hist(
        quark,
        label=["Quark Jet (Correct)", "Quark Jet (Incorrect)"],
        color=['skyblue', 'b'],
        stacked=True, bins=50, alpha=0.3)

    gluon_hists, _, _ = axarr[1].hist(
        gluon,
        label=["Gluon Jet (Correct)", "Gluon Jet (Incorrect)"],
        color=['lightcoral', 'r'],
        stacked=True, bins=50, alpha=0.5)

    max_value = max(each.max() for each in quark_hists + gluon_hists)
    y_max = 1.05 * max_value

    axarr[0].set_ylim(0, y_max)
    axarr[1].set_ylim(0, y_max)

    axarr[0].set_xlabel(uncertainty_name.title())

    for ax in axarr:
        ax.grid()
        ax.legend(fontsize=15)
    fig.savefig("./Plots/{}_{}.png".format(tag, uncertainty_name.replace(" ", "_")))
    plt.close(fig)

def draw_curve(is_sig, is_bkg,
               is_true_pos, is_false_pos,
               uncertainty, uncertainty_name,
               tag, title):
    signal_efficiency = []
    background_efficiency = []
    for cut in np.linspace(uncertainty.min(), uncertainty.max()):
        good = uncertainty < cut
        
        if sum(good) == 0:
            continue
        
        sig_eff = sum(is_true_pos[good]) / sum(is_sig)
        bkg_eff = sum(is_false_pos[good]) / sum(is_bkg)
        
        signal_efficiency.append(sig_eff)
        background_efficiency.append(bkg_eff)

    sig_eff = np.array(signal_efficiency)
    bkg_eff = np.array(background_efficiency)

    fig, ax = plt.subplots(figsize=(12, 8))
    suptitle = fig.suptitle(title, fontsize="x-large")

    # NOTE
    safe_idx = bkg_eff.nonzero()
    bkg_eff = bkg_eff[safe_idx]
    sig_eff = sig_eff[safe_idx]

    inv_sqrt_bkg_eff = 1 / np.sqrt(bkg_eff)
    line1 = ax.plot(sig_eff, inv_sqrt_bkg_eff,
                    marker="o", ls="--", color="mediumorchid",
                    label=r"$1 / \sqrt{\epsilon_{B}}$")
    ax.set_xlabel(r"$\epsilon_{S}$", fontdict={'size': 20})
    ax.set_ylabel(r"$1 / \sqrt{\epsilon_{B}}$", fontdict={'size': 20})
    # ax.grid()

    # NOTE
    twinx = ax.twinx()

    significance = sig_eff / np.sqrt(bkg_eff)
    line2 = twinx.plot(sig_eff, significance,
                       marker="^", ls="--", color="darkgreen",
                       label=r"$\epsilon_{S} / \sqrt{\epsilon_{B}}$")
    twinx.set_xlabel(r"$\epsilon_{S}$", fontdict={'size': 20})
    twinx.set_ylabel(r"$\epsilon_{S} / \sqrt{\epsilon_{B}}$", fontdict={'size': 20})
    # twinx.grid()

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, fontsize=20, loc='center right')


    fig.savefig("./Plots/{}_{}_significance.png".format(tag, uncertainty_name))
    plt.close(fig)




def draw(path, tag, title):
    print(path)

    npz_file = np.load(path)

    y_true = npz_file['y_true']
    prob_samples = npz_file['prob_samples']
    pred_samples = npz_file['pred_samples']

    y_pred = pred_samples.mean(axis=0).argmax(axis=1)

    variation_ratio = compute_variation_ratio(prob_samples)
    predictive_entropy = compute_predictive_entropy(prob_samples)
    mutual_information = compute_mutual_information(prob_samples)

    is_sig = y_true == 1
    is_bkg = y_true == 0

    is_true_pos = np.logical_and(is_sig, y_pred == 1)
    is_false_neg = np.logical_and(is_sig, y_pred == 0)

    is_true_neg = np.logical_and(is_bkg, y_pred == 0)
    is_false_pos = np.logical_and(is_bkg, y_pred == 1)

    draw_uncertainty(is_true_pos, is_false_neg,
                     is_true_neg, is_false_pos,
                     uncertainty=variation_ratio,
                     uncertainty_name='variation-ratio',
                     tag=tag,   
                     title=title)

    draw_uncertainty(is_true_pos, is_false_neg,
                     is_true_neg, is_false_pos,
                     uncertainty=predictive_entropy,
                     uncertainty_name='predictive entropy',
                     tag=tag,   
                     title=title)

    draw_uncertainty(is_true_pos, is_false_neg,
                     is_true_neg, is_false_pos,
                     uncertainty=mutual_information,
                     uncertainty_name='mutual information',
                     tag=tag,   
                     title=title)

    draw_curve(is_sig, is_bkg,
               is_true_pos, is_false_pos,
               uncertainty=variation_ratio,
               uncertainty_name='variation-ratio',
               tag=tag,
               title=title)

    draw_curve(is_sig, is_bkg,
               is_true_pos, is_false_pos,
               uncertainty=predictive_entropy,
               uncertainty_name='predictive entropy',
               tag=tag,
               title=title)

    draw_curve(is_sig, is_bkg,
               is_true_pos, is_false_pos,
               uncertainty=mutual_information,
               uncertainty_name='mutual information',
               tag=tag,
               title=title)
def main():
    draw("./logs/bnn_pt-100-110.npz",
         "pt-100-110",
         r'Dijet, $p_{T} \in (100, 110)$ GeV')

    draw("./logs/bnn_pt-200-220.npz",
         "pt-200-220",
         r'Dijet, $p_{T} \in (200, 220)$ GeV')

    draw("./logs/bnn_pt-500-550.npz",
         "pt-500-550",
         r'Dijet, $p_{T} \in (500, 550)$ GeV')


    draw("./logs/bnn_pt-1000-1100.npz",
         "pt-1000-1100",
         r'Dijet, $p_{T} \in (1000, 1100)$ GeV')




if __name__ == '__main__':
    main()
