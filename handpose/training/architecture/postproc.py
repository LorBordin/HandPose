from tensorflow.keras.models import Model
import tensorflow.keras.layers as  L
import tensorflow as tf

from .custom_layers import ExtractCoordinates
from .custom_layers import get_inverse_dist_grid
from .custom_layers import get_max_mask
from .custom_layers import grid_coords 

def create_postproc_model(grid_size, num_joints, head_type="k_heatmaps", name="post_processing"):
    """
        Returns a model to decode the movenet predictions. According to 'head_type' it performs the 
        following post-processing steps:

            - Keipoints Heatmaps:
                
                i.  Give a coase estimate of the keypoints coordinates by looking a the maximum of 
                    the corresponding heatmap.
             
                ii. Add the coordinates offset to the coords prediction to get a more refined
                    estimate.

                iii.Get an estimate of the keypoint visibility.

            - Center Heatmap: 

                i.  Give an estimate of the hand' center coordinates by looking a the maximum of 
                    the corresponding heatmap.
                ii. Predict the keypoint position coordinates by adding the keypoints offset coords 
                    to the center coords.
       
    Parameters
    ----------
    grid_size: int 
        Size of the heatmap.
    num_joints: int
        Number of predicted keypoints.
    head_type: str
            Type of model's head. Options: 'k_heatmaps', 'c_heatmap'. By default 'k_heatmap'.
    name: str
        Model name. The default value is "post_processing".
    """     

    if head_type == "k_heatmaps":

        heatmaps = L.Input((grid_size, grid_size, num_joints))
        offsets_grid = L.Input((grid_size, grid_size, num_joints * 2))

        inputs = [heatmaps, offsets_grid]

        # 1. Get the coordinates of the keypoints 
        kpts_mask = L.Lambda(lambda x: get_max_mask(x))(heatmaps)
        coords = ExtractCoordinates(n_rep=1)(kpts_mask)

        # 2. Get the offsets from the keypoints heatmaps
        offset_mask = L.Concatenate()([kpts_mask]*2)
        offsets_grid = L.Multiply()([offsets_grid, offset_mask])
        offsets_grid = L.GlobalMaxPooling2D()(offsets_grid)  
        offsets_grid = L.Lambda(lambda x: x / tf.cast(grid_size, tf.float32))(offsets_grid)

        # 3. Get a refined prediction of the keypoints coordinates 
        coords = L.Add(name="final_coords")([coords, offsets_grid])
        coords = L.Reshape((2, -1))(coords)
        coords = L.Permute((2,1))(coords)
    
        # 4. Get the keypoints probability
        probas = L.Multiply()([heatmaps, kpts_mask])
        probas = L.GlobalMaxPooling2D()(probas)
        probas = L.Reshape((-1, 1))(probas)

    elif head_type == "c_heatmap":
        
        heatmaps = L.Input((grid_size, grid_size, 1))
        offsets_grid = L.Input((grid_size, grid_size, num_joints * 2))
        probas_grid = L.Input((grid_size, grid_size, num_joints))

        inputs = [heatmaps, offsets_grid, probas_grid]

        # 1. Get the coordinates of the hand center 
        center_mask = L.Lambda(lambda x: get_max_mask(x))(heatmaps)
        center = ExtractCoordinates(n_rep=num_joints)(center_mask)

        # 2. Get the keypoints offset from the hand's center 
        offset_mask = L.Concatenate()([center_mask]*num_joints*2)
        
        k_offsets = L.Multiply()([offsets_grid, offset_mask]) 
        k_offsets = L.GlobalAveragePooling2D()(k_offsets) 
        k_offsets = L.Lambda(lambda x: x * grid_size * grid_size)(k_offsets)
    
        # 3. Get the keypoints coordinates
        coords = L.Add()([k_offsets, center])
        coords = L.Reshape((2, -1))(coords)
        coords = L.Permute((2,1))(coords)

        # 4. Get the keypoint probability
        probas_mask = L.Concatenate()([center_mask]*num_joints)

        probas = L.Multiply()([probas_grid, probas_mask])
        probas = L.GlobalMaxPooling2D()(probas)
        probas = L.Reshape((-1, 1))(probas)
    
    # 7. Concatenate the outputs together
    coords = L.Concatenate()([probas, coords])

    post_proc = Model(inputs, [coords, heatmaps], name=name)
    
    return post_proc
