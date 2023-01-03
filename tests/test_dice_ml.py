# import dice_ml
# from dice_ml.utils import helpers # helper functions
# from sklearn.model_selection import train_test_split

# dataset = helpers.load_adult_income_dataset()
# target = dataset["income"] # outcome variable
# train_dataset, test_dataset, _, _ = train_test_split(dataset,
#                                                      target,
#                                                      test_size=0.2,
#                                                      random_state=0,
#                                                      stratify=target)
# # Dataset for training an ML model
# d = dice_ml.Data(dataframe=train_dataset,
#                  continuous_features=['age', 'hours_per_week'],
#                  outcome_name='income')

# # Pre-trained ML model
# m = dice_ml.Model(model_path=dice_ml.utils.helpers.get_adult_income_modelpath(),
#                   backend='TF2', func="ohe-min-max")
# # DiCE explanation instance
# exp = dice_ml.Dice(d,m)

# # Generate counterfactual examples
# query_instance = test_dataset.drop(columns="income")[0:1]
# dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite")
# # Visualize counterfactual explanation
# dice_exp.visualize_as_dataframe()

# print(dice_exp)