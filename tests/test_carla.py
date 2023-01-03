# from carla import OnlineCatalog, MLModelCatalog
# from carla.recourse_methods import GrowingSpheres

# # # load a catalog dataset
# # data_name = "adult"
# # dataset = OnlineCatalog(data_name)

# # # load artificial neural network from catalog
# # model = MLModelCatalog(dataset, "ann")

# # get factuals from the data to generate counterfactual examples
# factuals = dataset.raw.iloc[:10]

# # load a recourse model and pass black box model
# gs = GrowingSpheres(model)

# # generate counterfactual examples
# counterfactuals = gs.get_counterfactuals(factuals)