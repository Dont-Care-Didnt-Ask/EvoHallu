"""Render fixed-case normalization diagnostics from complete saved JSON only."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/overnight20261021'
CASES=['original_128','original_512','original_2048','original_full','shuffle451_512']
COMMON='geometry__common_claim_true_false'

def main():
    batch_path=RESULTS/'batch_sensitivity/BATCH_SENSITIVITY.json'
    train_path=RESULTS/'frozen_train_normalization/FROZEN_NORMALIZATION_REPORT.json'
    batch=json.loads(batch_path.read_text());train=json.loads(train_path.read_text())
    assert batch['status']=='complete_fixed_five_case_diagnostic' and train['status']=='complete_exploratory_train_only_control'
    assert batch['original512_byte_parity_all9'] and train['all9_original_proxy_primary_exact'] and train['all36_frozen_scores_chunk_order_byte_invariant']
    assert batch['source_sha256']==train['parent_source_sha256']
    macros={r['case']:r for r in batch['macros']};assert set(macros)==set(CASES)
    assert all(macros[c]['datasets']==9 for c in CASES)
    cc={r['case']:r for r in batch['per_dataset'] if r['dataset']==COMMON};assert set(cc)==set(CASES)
    fixed_cc=next(r for r in train['per_dataset'] if r['dataset']==COMMON)
    assert len(train['per_dataset'])==9 and fixed_cc['rows']==4450
    assert macros['original_512']['paper_macro_auc_percent']==train['macro']['original512_folded']
    assert cc['original_512']['paper_auc_percent']==fixed_cc['original512_folded']
    panel_a=[macros[c]['paper_macro_auc_percent'] for c in CASES]+[train['macro']['frozen_folded']]
    panel_b=[cc[c]['paper_auc_percent'] for c in CASES]+[fixed_cc['frozen_folded']]
    references=[train['macro']['nll_folded'],fixed_cc['nll_folded']]

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'xtick.labelsize':8,'ytick.labelsize':8,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.9),gridspec_kw={'wspace':.25})
    sensitivity='#376b9e';primary='#cc6b18';fixed='#72549b';nll='#5b6066'
    labels=['128','512\n(primary)','2,048','Full','Shuffled\n512','Train-fixed\nstatistics']
    for i,(ax,values,reference) in enumerate(zip(axes,[panel_a,panel_b],references)):
        ax.axvspan(4.55,5.5,color=fixed,alpha=.065,lw=0)
        ax.axvline(4.55,color='#c9c5ce',lw=.8)
        ax.axhline(reference,color=nll,lw=1.1,ls=(0,(4,3)),zorder=1)
        for j,value in enumerate(values):
            color=primary if j==1 else fixed if j==5 else sensitivity
            marker='D' if j==1 else 's' if j==5 else 'o'
            ax.scatter(j,value,s=58 if j in (1,5) else 43,color=color,marker=marker,edgecolor='white',linewidth=.7,zorder=3)
            ax.annotate(f'{value:.2f}',(j,value),xytext=(0,8),textcoords='offset points',ha='center',va='bottom',fontsize=8,color=color,
                        fontweight='bold' if j in (1,5) else 'normal')
        ax.set_xlim(-.5,5.5);ax.set_xticks(np.arange(6),labels)
        ax.grid(axis='y',color='#dddddd',lw=.6,alpha=.8);ax.set_axisbelow(True)
        ax.set_xlabel('Chunk size / order, or separate normalization control',labelpad=9)
        ax.text(.015,.025,f'Canonical NLL: {reference:.2f}',transform=ax.transAxes,color=nll,fontsize=8,va='bottom',
                bbox=dict(facecolor='white',edgecolor='none',pad=1,alpha=.9))
    axes[0].set_ylim(67.9,71.75);axes[0].set_ylabel('AUROC × 100 (dataset-wise fold)')
    axes[1].set_ylim(48.0,71.7);axes[1].set_ylabel('AUROC × 100')
    axes[0].set_title('a  Source benchmark: nine-dataset mean',loc='left',pad=12)
    axes[1].set_title('b  Common Claims: 4,450 rows',loc='left',pad=12)
    handles=[Line2D([],[],marker='o',color='none',markerfacecolor=sensitivity,markeredgecolor='white',markersize=7,label='Sensitivity case'),
             Line2D([],[],marker='D',color='none',markerfacecolor=primary,markeredgecolor='white',markersize=7,label='Original 512 primary'),
             Line2D([],[],marker='s',color='none',markerfacecolor=fixed,markeredgecolor='white',markersize=7,label='Train-fixed control (5,605 fit rows)'),
             Line2D([],[],color=nll,ls=(0,(4,3)),lw=1.1,label='Canonical NLL')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.185),ncol=4,frameon=False,fontsize=8,columnspacing=1.25,handletextpad=.5)
    fig.text(.065,.135,'Common Claims order: 2,225 label-0 rows followed by 2,225 label-1 rows. No evaluation labels enter normalization or shuffling.',fontsize=8,color='#333333')
    fig.text(.065,.087,'Original training, validation and benchmark scoring all used 512-row chunks. Other settings are diagnostics; the original primary is retained.',fontsize=7.7,color='#444444')
    fig.text(.065,.045,'Point estimates only. The train-fixed result is a separate exploratory control designed after test viewing; all fixed cases are shown.',fontsize=7.7,color='#444444')
    fig.suptitle('Frozen compositional winner (seed 8741): normalization diagnostics',fontsize=12,y=.99)
    fig.subplots_adjust(left=.065,right=.985,top=.84,bottom=.365)
    output=RESULTS/'figures';output.mkdir(parents=True,exist_ok=True);stem=output/'batch_normalization_diagnostics'
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (batch_path,train_path)}
    description=json.dumps({'source_json_sha256':hashes,'source9_folded':panel_a,'common_claims_auc':panel_b,
                             'nll_references':references,'case_order':CASES+['separate_train_fixed_control'],'primary_replaced':False})
    fig.savefig(str(stem)+'.png',dpi=300,metadata={'Description':description,'Title':'Frozen-program normalization diagnostics'})
    fig.savefig(str(stem)+'.pdf',metadata={'Title':'Frozen-program normalization diagnostics','Subject':description,'Creator':'EvoHallu plot_batch_normalization_diagnostics.py'})
    plt.close(fig)
    print(json.dumps({'png':str(stem)+'.png','pdf':str(stem)+'.pdf','values_checked_from_saved_json':True,'source9_folded':panel_a,'common_claims_auc':panel_b,'nll_references':references}))

if __name__=='__main__':main()
