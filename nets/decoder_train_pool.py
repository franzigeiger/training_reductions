from brainscore.submission.utils import UniqueKeyDict

brain_translated_pool = UniqueKeyDict()

# for identifier in trained_models.keys():
#     for epoch in (
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 43, 37, 26, 50, 52):
#         brain_translated_pool[f'{identifier}_BF_epoch_{epoch:02d}'] = LazyLoad(
#             lambda id=identifier, e=epoch: cornet_s_brainmodel(f'{id}_BF_epoch_{e:02d}', True))
#
# for identifier in v4.keys():
#     for epoch in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 43, 37, 26, 50, 52):
#         brain_translated_pool[f'{identifier}_BF_epoch_{epoch:02d}'] = LazyLoad(
#             lambda id=identifier, e=epoch: cornet_s_brainmodel(f'{id}_BF_epoch_{e:02d}', True))
#
# for identifier in trained_models.keys():
#     for epoch in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 43, 37, 26, 50, 52):
#         brain_translated_pool[f'{identifier}_epoch_{epoch:02d}'] = LazyLoad(
#             lambda id=identifier, e=epoch: cornet_s_brainmodel(f'{id}_epoch_{e:02d}', True))
#
# for identifier in v4.keys():
#     for epoch in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 43, 37, 26, 50, 52):
#         brain_translated_pool[f'{identifier}_epoch_{epoch:02d}'] = LazyLoad(
#             lambda id=identifier, e=epoch: cornet_s_brainmodel(f'{id}_epoch_{e:02d}', True))for identifier in trained_models.keys():
# for identifier in trained_models.keys():
#     for epoch in (for identifier in trained_models.keys():):
#         brain_translated_pool[f'{identifier}_BF_epoch_{epoch:02d}'] = LazyLoad(
#             lambda id=identifier, e=epoch: cornet_s_brainmodel(f'{id}_BF_epoch_{e:02d}', True))
#
# for identifier in v4.keys():
#     for epoch in (for identifier in trained_models.keys():):
#         brain_translated_pool[f'{identifier}_BF_epoch_{epoch:02d}'] = LazyLoad(
#             lambda id=identifier, e=epoch: cornet_s_brainmodel(f'{id}_BF_epoch_{e:02d}', True))
#
# for identifier in trained_models.keys():
#     for epoch in (for identifier in trained_models.keys():):
#         brain_translated_pool[f'{identifier}_epoch_{epoch:02d}'] = LazyLoad(
#             lambda id=identifier, e=epoch: cornet_s_brainmodel(f'{id}_epoch_{e:02d}', True))
#
# for identifier in v4.keys():
#     for epoch in (for identifier in trained_models.keys():):
#         brain_translated_pool[f'{identifier}_epoch_{epoch:02d}'] = LazyLoad(
#             lambda id=identifier, e=epoch: cornet_s_brainmodel(f'{id}_epoch_{e:02d}', True))
#
# for epoch in (0, 1, 2, 3, 4, 5, 10, 15, 20, 90):
#     identifier = f'resnet_mil_trained_epoch_{epoch:02d}'
#     layers = LazyLoad(lambda: resnet_michael_layers())

# for identifier, activations_model in Hooks().iterate_hooks(basemodel_identifier, activations_model):
# if identifier in self:  # already pre-defined
#     continue
# from model_tools.brain_transformation import ModelCommitment
#
# activations_model = LazyLoad(lambda id=identifier, e=epoch: resnet_michael(id, True))
#
#
# # enforce early parameter binding: https://stackoverflow.com/a/3431699/2225200
# def load(identifier=identifier, activations_model=activations_model, layers=layers):
#     brain_model = ModelCommitment(identifier=identifier, activations_model=activations_model,
#                                   layers=layers)
#     for region in ['V1', 'V2', 'V4', 'IT']:
#         brain_model.commit_region(region)
#     return brain_model
#
#
# brain_translated_pool[identifier] = LazyLoad(load)
