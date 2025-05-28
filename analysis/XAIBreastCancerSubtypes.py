import xgboost as xgb
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# Read the dataframe consisting of breast cancer subtype gene expression
df = pd.read_csv(fiu_pickle) #Dependent on hospital data 

# Define the target column name
target_column = "Breast_Cancer_Subtype" 

# Split the data into train and test sets using StratifiedKFold
skf = StratifiedKFold(n_splits=5)
target = df[target_column]

fold_no = 1
for train_index, test_index in skf.split(df, target):
    train = df.loc[train_index, :]
    test = df.loc[test_index, :]
    train_filename = 'train_split_' + str(fold_no) + '.csv'
    test_filename = 'test_split_' + str(fold_no) + '.csv'
    train.to_csv(train_filename, index=False)
    test.to_csv(test_filename, index=False)
    fold_no += 1

# Define a function to drop falsely predicted samples by the model
def drop_false_pred(df):
    df = df[df.predicted_label.astype(int) == df.true_label.astype(int)].copy()
    df.drop("predicted_label", axis=1, inplace=True)
    return df

# Define a function to filter samples based on the true predicted label for calculating SHAP values
def filter_shap(test_data, shap_arr, y_map_new):
    df_data = []
    ids_list = test_data.index.to_list()
    genes_list = test_data.columns.tolist()
    genes_list = genes_list[0:19648]

    for i in tqdm(range(shap_arr.shape[0])):
        sample = shap_arr[i]
        sample_id = ids_list[i]
        label = y_map_new['pred'][i]
        label = map_dict[label]

        shap_scores_flat = sample[label][: len(genes_list)]

        df_data.append([sample_id, *list(shap_scores_flat), label])

    shap_df = pd.DataFrame(
        data=np.array(df_data), columns=["id", *genes_list, "predicted_label"]
    )

    shap_df.set_index("id", inplace=True)
    shap_df["true_label"] = list(y_map_new['true_label'])

    return shap_df

# Define a function to get the ranking of genes for each class based on SHAP values
def get_rank_df(df):
    df = drop_false_pred(df)
    shap_df = df.loc[:, df.columns != "true_label"].astype("float64")

    median_shap = shap_df.groupby("true_label").median()
    median_shap = median_shap.T

    rank_dict = {}
    for col in median_shap:
        sorted_df = median_shap.sort_values(by=col, ascending=False)
        rank_dict[col] = sorted_df.index
    rank_df = pd.DataFrame.from_dict(rank_dict)

    return rank_df

model_dir = ""
output_dir = ""

for i in range(1, 6):
    # Load the truly predicted test file
    newdftest = pd.read_csv(output_dir + "AfterDroppingFalsePrediction_test_" + str(i) + ".csv")
    newdftest.index = newdftest["Unnamed: 0"]
    newdftest = newdftest.drop(["Unnamed: 0"], axis="columns")
    X_test = newdftest.copy()

    # Load the truly predicted y_map
    y_map_new = pd.read_csv("true_predicted_label_test_" + str(i) + ".csv")
    y_map_new.index = y_map_new["Unnamed: 0"]
    y_map_new = y_map_new.drop(["Unnamed: 0"], axis='columns')

    # Load the saved model
    mcl = xgb.XGBRFClassifier()
    mcl.load_model(model_dir + "fold_" + str(i) + "_model_topology.json")

    # Tree explainer; mcl is the model of xgboost
    explainer = shap.TreeExplainer(mcl)

    # Calculating SHAP values
    out_list = []
    num_samples = np.shape(X_test)[0]
    for sample in tqdm(range(0, num_samples)):
        shap_values = explainer.shap_values(X_test[sample : sample + 1])
        out_list.append(shap_values)

    shap_arr = np.squeeze(np.array(out_list))

    shap_df = filter_shap(X_test, shap_arr, y_map_new)

    # Store SHAP values
    output_file = output_dir + "treeExplainer_shap_scores_allGenes_test_" + str(i) + ".pkl"
    pickle.dump(shap_df, open(str(output_file), "wb"))

    # Class-specific gene ranking
    rank_df = get_rank_df(shap_df)
    rank_df.columns = ["healthy", "luad", "lusc"]
    rank_df.to_csv(output_dir + "class_specific_gene_rank_test_" + str(i) + ".csv")

    shap.summary_plot(explainer.shap_values(X_test), X_test, plot_type="bar", class_names=rank_df.columns, feature_names=X_test.columns, show=False)
    plt.savefig(output_dir + "global_interpretation_top_gene_rank_test_" + str(i) + ".png")
    plt.clf()

    # Summary plot specific class healthy
    shap.summary_plot(shap.TreeExplainer(mcl).shap_values(X_test)[0], X_test.values, feature_names=X_test.columns, show=False)
    plt.savefig(output_dir + "class_specific_healthy_interpretation_top_gene_rank_test_" + str(i) + ".png")
    plt.clf()

    # Summary plot specific class luad
    shap.summary_plot(shap.TreeExplainer(mcl).shap_values(X_test)[1], X_test.values, feature_names=X_test.columns, show=False)
    plt.savefig(output_dir + "class_specific_luad_interpretation_top_gene_rank_test_" + str(i) + ".png")
    plt.clf()

    # Summary plot specific class lusc
    shap.summary_plot(shap.TreeExplainer(mcl).shap_values(X_test)[2], X_test.values, feature_names=X_test.columns, show=False)
    plt.savefig(output_dir + "class_specific_lusc_interpretation_top_gene_rank_test_" + str(i) + ".png")
    plt.clf()

# Patient Specific Genes
for j in range(1, 6):
    with open("treeExplainer_shap_scores_allGenes_test_" + str(j) + ".pkl", "rb") as f:
        shap_scores = pickle.load(f)
    features_shap_df = shap_scores.drop(['predicted_label', 'true_label'], axis="columns")

    genelist = []
    for i in range(0, features_shap_df.shape[0]):
        s = pd.DataFrame(features_shap_df.iloc[i])
        s = s.sort_values(by=features_shap_df.index[i], ascending=False)
        genelist.append(s.index.tolist())

    patientSpecificGenes = pd.DataFrame(genelist)
    patientSpecificGenes.index = features_shap_df.index
    patientSpecificGenes['true_label'] = shap_df['true_label']
    patientSpecificGenes.to_csv("Patient_specific_genes_xgboost_treeExplainer_test_" + str(j) + ".csv")

# Forming Single Dataframe for patient specific genes
output_dir = ""
df = pd.DataFrame()
for i in tqdm(range(1, 6)):
    df_1 = pd.read_csv(output_dir + "Patient_specific_genes_xgboost_treeExplainer_test_" + str(i) + ".csv")
    df = pd.concat([df, df_1], axis=0)

y_map_all = pd.DataFrame()
for i in range(1, 6):
    ymap = pd.read_csv(output_dir + "true_predicted_label_test_" + str(i) + ".csv")
    y_map_all = pd.concat([y_map_all, ymap], axis=0)

y_map_all.index = y_map_all["Unnamed: 0"]
new_y_map_all = y_map_all.drop("Unnamed: 0", axis="columns")
new_y_map_all = new_y_map_all[['class']]

patientSpecificGenes = pd.merge(df, new_y_map_all, left_on="id", right_on=new_y_map_all.index)
patientSpecificGenes = patientSpecificGenes.drop("true_label", axis="columns")
patientSpecificGenes
