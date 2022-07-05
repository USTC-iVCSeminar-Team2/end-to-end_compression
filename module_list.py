import torch


def compressor_list(a, h):
    if a.model_name == 'image_compressor':
        from models.image_compressor import ImageCompressor
        model = ImageCompressor(h)
        print('Successfully load model: {}'.format(a.model_name))
        return model
    else:
        raise Exception('Cannot find model: {}'.format(a.model_name))


def optimizer_list(model, h):
    if h.optim_name == 'AdamW':
        optim = torch.optim.AdamW(model.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
        return optim