from __future__ import print_function
import argparse
import pandas as pd
from itertools import product, permutations, combinations, combinations_with_replacement
import numpy as np

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate experimental scRNA-seq inputs.', epilog = 'Example usage to generate dataset with all TFs and 500 genes (TFs+500): python generateInputs.py -e=ExpressionData.csv -g=GeneOrdering.csv -f=STRING-network.csv -i=human-tfs.csv -p=0.01 -c -t -n=500 -o=temp')
    parser.add_argument('-e','--expFile', type = str)
    parser.add_argument('-g','--geneOrderingFile', type = str)
    parser.add_argument('-f','--netFile', type = str)
    parser.add_argument('-i','--TFFile', type = str)
    parser.add_argument('-p','--pVal', type=float)
    parser.add_argument('-c','--BFcorr', action='store_true')
    parser.add_argument('-n','--numGenes', type=int)
    parser.add_argument('-t','--TFs', action='store_true')
    parser.add_argument('-o','--outPrefix', type = str, default = 'BL-')
    return parser
def parse_arguments():
    parser = get_parser()
    opts = parser.parse_args()
    return opts

opts = parse_arguments()
include_tfs = opts.TFs
expr_file = opts.expFile
gene_ordering_file = opts.geneOrderingFile
tf_file = opts.TFFile
pval_cutoff = opts.pVal
bf_corr = opts.BFcorr
num_genes = opts.numGenes
sort_by_variance = True
print("\nReading %s" % (expr_file))
expr_df = pd.read_csv(expr_file, header=0, index_col=0)
print("\nReading %s" % (gene_ordering_file))
gene_df = pd.read_csv(gene_ordering_file, header=0, index_col=0)
if include_tfs:
    print("\nReading %s" % (tf_file))
    tfs_df = pd.read_csv(tf_file, header=0)
    tfs = tfs_df[tfs_df.columns[0]]
    num_total_tfs = len(tfs)
    # limit the tfs to those present in the expression file
    tfs = tfs[tfs.isin(expr_df.index)]
    print("\t%d/%d TFs present in ExpressionData" % (
        len(tfs), num_total_tfs))
expr_variable_genes = set(gene_df.index.values) & set(expr_df.index.values)
extra_genes = set(gene_df.index.values) - set(expr_df.index.values)
if len(extra_genes) != 0:
    print("\nWARNING: %d variable genes are not in ExpressionData.csv:" % (len(extra_genes)))
    print(extra_genes)
    gene_df = gene_df.loc[expr_variable_genes]
# limit the Expression matrix to the variable genes
pval_col = gene_df.columns[0]
# make sure its sorted by the pvalue column
gene_df.sort_values(by=pval_col, inplace=True)
variable_genes = gene_df.index.values
if pval_cutoff !=0 :
    if bf_corr:
        pval_cutoff = pval_cutoff / float(len(gene_df.index))
        print("\nUsing the BF-corrected p-value cutoff of %s (%s / %s genes)" % (
            pval_cutoff, pval_cutoff*float(len(gene_df.index)), len(gene_df.index)))

    variable_genes = gene_df[gene_df[pval_col] < pval_cutoff].index.values
    print("\n%d genes pass pval_cutoff of %s" % (len(variable_genes), pval_cutoff))
    #print("\nBefore using pValue cut-off num rows: ", gene_df.shape[0] )
    gene_df = gene_df.filter(items = variable_genes, axis='index')
    print("\nAfter using pValue cut-off num rows: ", gene_df.shape[0] )

variable_genes = []
if include_tfs:
    # include the TFs that pass the p-val cutoff
    tfs = tfs[tfs.isin(gene_df.index)]
    if pval_cutoff:
        print("\nIncluding %d TFs that pass the pval cutoff" % (len(tfs)))
    else:
        print("\nIncluding %d TFs" % (len(tfs)))
    variable_tfs = set(tfs)
    gene_df.drop(labels = variable_tfs, axis='index', inplace = True)
else:
    variable_tfs = set()
if num_genes > 0:
    if num_genes > len(gene_df):
        variable_genes_new = gene_df.index
        pass
    else:
        if sort_by_variance:
            if len(gene_df.columns) < 2:
                print("ERROR: no variance column found. Should be 3rd column. Quitting")
                sys.exit()
            var_col = gene_df.columns[1]
            gene_df.sort_values(by=var_col, inplace=True, ascending = False)
        variable_genes_new = gene_df.iloc[:num_genes].index.values
    variable_genes = set(variable_genes_new) | set(variable_tfs)
print("\nRestricting to %d genes" % (len(variable_genes)))
expr_df = expr_df.loc[variable_genes]
print("\nNew shape of Expression Data %d x %d" % (expr_df.shape[0],expr_df.shape[1]))
expr_df.to_csv(opts.outPrefix+'-ExpressionData.csv')
if opts.netFile != 'None':
    netFile = opts.netFile
    netDF = pd.read_csv(netFile)
    netDF = netDF[(netDF.Gene1.isin(expr_df.index)) & (netDF.Gene2.isin(expr_df.index))]
    netDF = netDF[netDF.Gene1 != netDF.Gene2]
    netDF.drop_duplicates(keep = 'first', inplace=True)
    netDF.to_csv(opts.outPrefix+'-network.csv', index=False)
    allNodes = set(netDF.Gene1.unique()).union(set(netDF.Gene2.unique()))
    nTFs = expr_df[expr_df.index.isin(netDF.Gene1.unique())].shape[0]
    nGenes = expr_df[expr_df.index.isin(allNodes)].shape[0]
    print("\n#TFs: %d, #Genes: %d, #Edges: %d, Density: %.3f" % (nTFs,nGenes,netDF.shape[0],netDF.shape[0]/((nTFs*nGenes)-nTFs)))

print("\n\nExiting...\n")
