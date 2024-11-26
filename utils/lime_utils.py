import numpy as np
import torch
import torchvision


# def get_probab_class(tensor, model, target_class=15, num_classes=2):
#     results = batch_predict(tensor, model)
#     one_hot_results = []
#     _length_dataset = num_classes
    
#     # for all images
#     for _i,result in enumerate(results):

#         _enter_aux = False
#         # generate a one-hot encoding
#         one_hot = np.zeros(_length_dataset)
        
#         for box in result.boxes:
#             _class_index = int(box.cls.item())
        
#             if _class_index == target_class:
#                 one_hot[_class_index] = 1
#                 _enter_aux = True
                
#         if _enter_aux is False:
#             one_hot = np.full(_length_dataset,1/_length_dataset)
        
#         one_hot_results.append(one_hot)
            
#     return np.array(one_hot_results)

def get_probab_class_wrapper(img_np, model, target_class=15, num_classes=2):
    """
    Wrapper function to create a callable for LimeImageExplainer
    that accepts only the images argument, fixing target_class and conf_th.
    """
    def get_probab_class(img_np=img_np, model=model, target_class=target_class, num_classes=num_classes):
        with torch.no_grad():        
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
            batch = torch.stack([transform(img) for img in img_np], dim=0)
            results = model.predict(batch, verbose=False)

        probab_results = []
        _length_dataset = num_classes
        
        # for all images
        for _i,result in enumerate(results):

            # generate a one-hot encoding
            conf_scores = []
            
            for box in result.boxes:
                _class_index = int(box.cls.item())
            
                if _class_index == target_class:
                    conf_scores.append( float(box.conf.item()) )
                
            # from confidence scores to probabilities
            if conf_scores:
                prob_cls = np.mean(conf_scores)
                other_probs = (1-prob_cls)/(_length_dataset-1)
                probability_vector = np.full(_length_dataset, other_probs)
                probability_vector[target_class] = prob_cls
            else:
                probability_vector = np.full(_length_dataset,1/_length_dataset)
            
            probab_results.append(probability_vector)
                
        return np.array(probab_results)
    return get_probab_class  # Return the customized function