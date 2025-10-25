'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
''' 

import torch
import os
import pandas
import numpy as np
import tqdm
import glob
import sys
import yaml
import logging
from PIL import Image, ImageFile

# Allow PIL to load truncated images (handles partially corrupted JPEG files)
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize
from utils.fusion import apply_fusion
from networks import create_architecture, load_weights

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']


def runnig_tests(input_csv, weights_dir, models_list, device, batch_size=1, output_csv=None, save_interval=100):
    """
    Run tests on images with specified models.
    
    Args:
        input_csv: Path to input CSV file
        weights_dir: Directory containing model weights
        models_list: List of model names to test
        device: Torch device to use
        batch_size: Batch size for processing
        output_csv: Path to output CSV file (for periodic saving)
        save_interval: Number of samples between saves
    """
    table = pandas.read_csv(input_csv)[['filename',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    
    # Initialize columns for each model with NaN
    for model_name in models_list:
        table[model_name] = np.nan
    
    # Create output CSV file at start if specified
    if output_csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        table.to_csv(output_csv, index=False)
        logging.info(f"Initialized output CSV: {output_csv}")
    
    # Store model configs instead of loading all models at once
    model_configs = dict()
    transform_dict = dict()
    print("Preparing models:")
    for model_name in models_list:
        try:
            print(model_name, flush=True)
            _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)

            # Create transform but don't load model yet
            transform = list()
            if patch_size is None:
                print('input none', flush=True)
                transform_key = 'none_%s' % norm_type
            elif patch_size=='Clip224':
                print('input resize:', 'Clip224', flush=True)
                transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
                transform.append(CenterCrop((224, 224)))
                transform_key = 'Clip224_%s' % norm_type
            elif isinstance(patch_size, tuple) or isinstance(patch_size, list):
                print('input resize:', patch_size, flush=True)
                transform.append(Resize(*patch_size))
                transform.append(CenterCrop(patch_size[0]))
                transform_key = 'res%d_%s' % (patch_size[0], norm_type)
            elif patch_size > 0:
                print('input crop:', patch_size, flush=True)
                transform.append(CenterCrop(patch_size))
                transform_key = 'crop%d_%s' % (patch_size, norm_type)
            
            transform.append(make_normalize(norm_type))
            transform = Compose(transform)
            transform_dict[transform_key] = transform
            
            # Store config instead of loading model
            model_configs[model_name] = {
                'transform_key': transform_key,
                'model_path': model_path,
                'arch': arch
            }
            print(flush=True)
        except Exception as e:
            logging.error(f"Failed to prepare model {model_name}: {e}")
            # Remove the column for failed model
            table = table.drop(columns=[model_name])
            continue

    ### test
    with torch.no_grad():
        
        do_models = list(model_configs.keys())
        do_transforms = set([model_configs[_]['transform_key'] for _ in do_models])
        print(do_models)
        print(do_transforms)
        print(flush=True)
        
        if not do_models:
            logging.error("No models loaded successfully. Exiting.")
            return table
        
        print("Running the Tests")
        
        # Process each model separately to save memory
        for model_name in do_models:
            logging.info(f"Processing with model: {model_name}")
            
            # Load only the current model
            try:
                config = model_configs[model_name]
                model = load_weights(create_architecture(config['arch']), config['model_path'])
                model = model.to(device).eval()
            except Exception as e:
                logging.error(f"Failed to load model {model_name}: {e}")
                continue
            
            batch_img = list()
            batch_id = list()
            last_index = table.index[-1]
            processed_count = 0
            failed_images = []
            
            for index in tqdm.tqdm(table.index, total=len(table), desc=f"Model {model_name}"):
                filename = os.path.join(rootdataset, table.loc[index, 'filename'])
                
                # Try to load and transform the image
                try:
                    img = Image.open(filename).convert('RGB')
                    batch_img.append(transform_dict[config['transform_key']](img))
                    batch_id.append(index)
                except FileNotFoundError:
                    if filename not in failed_images:
                        logging.warning(f"File not found: {filename}")
                        failed_images.append(filename)
                    table.loc[index, model_name] = np.nan
                    continue
                except Exception as e:
                    if filename not in failed_images:
                        logging.warning(f"Error loading/transforming {filename}: {e}")
                        failed_images.append(filename)
                    table.loc[index, model_name] = np.nan
                    continue

                if (len(batch_id) >= batch_size) or (index==last_index):
                    try:
                        batch_tensor = torch.stack(batch_img, 0)
                        out_tens = model(batch_tensor.to(device)).cpu().numpy()

                        if out_tens.shape[1] == 1:
                            out_tens = out_tens[:, 0]
                        elif out_tens.shape[1] == 2:
                            out_tens = out_tens[:, 1] - out_tens[:, 0]
                        else:
                            logging.error(f"Unexpected output shape for {model_name}: {out_tens.shape}")
                            batch_img = list()
                            batch_id = list()
                            continue
                        
                        if len(out_tens.shape) > 1:
                            logit1 = np.mean(out_tens, (1, 2))
                        else:
                            logit1 = out_tens

                        for ii, logit in zip(batch_id, logit1):
                            table.loc[ii, model_name] = logit
                        
                        processed_count += len(batch_id)
                        
                        # Periodically save results
                        if output_csv is not None and processed_count % save_interval == 0:
                            table.to_csv(output_csv, index=False)
                            logging.info(f"Progress saved: {processed_count}/{len(table)} samples processed for {model_name}")
                    
                    except Exception as e:
                        logging.error(f"Error processing batch with model {model_name}: {e}")
                        # Mark batch as failed for this model
                        for ii in batch_id:
                            table.loc[ii, model_name] = np.nan
                    
                    batch_img = list()
                    batch_id = list()
            
            # Free GPU memory after each model
            del model
            torch.cuda.empty_cache()
            
            # Save after each model completes
            if output_csv is not None:
                table.to_csv(output_csv, index=False)
                logging.info(f"Model {model_name} completed. Results saved.")
        
        # Final save
        if output_csv is not None:
            table.to_csv(output_csv, index=False)
            logging.info(f"Final results saved to {output_csv}")
        
        if failed_images:
            logging.warning(f"Total failed images: {len(failed_images)}")
            logging.info(f"First few failed images: {failed_images[:10]}")
        
    return table


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"       , '-i', type=str, help="The path of the input csv file with the list of images")
    parser.add_argument("--out_csv"      , '-o', type=str, help="The path of the output csv file", default="./results.csv")
    parser.add_argument("--weights_dir"  , '-w', type=str, help="The directory to the networks weights", default="./weights")
    parser.add_argument("--models"       , '-m', type=str, help="List of models to test", default='clipdet_latent10k_plus,Corvi2023')
    parser.add_argument("--fusion"       , '-f', type=str, help="Fusion function", default='soft_or_prob')
    parser.add_argument("--device"       , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--save_interval", '-s', type=int, help="Number of samples between saves", default=100)
    args = vars(parser.parse_args())
    
    if args['models'] is None:
        args['models'] = os.listdir(args['weights_dir'])
    else:
        args['models'] = args['models'].split(',')
    
    try:
        # Pass output_csv and save_interval to enable periodic saving
        table = runnig_tests(
            args['in_csv'], 
            args['weights_dir'], 
            args['models'], 
            args['device'],
            output_csv=args['out_csv'],
            save_interval=args['save_interval']
        )
        
        if args['fusion'] is not None:
            try:
                # Only apply fusion to models that successfully loaded
                available_models = [m for m in args['models'] if m in table.columns]
                if available_models:
                    table['fusion'] = apply_fusion(table[available_models].values, args['fusion'], axis=-1)
                else:
                    logging.error("No models available for fusion")
            except Exception as e:
                logging.error(f"Error applying fusion: {e}")
        
        # Final save
        output_csv = args['out_csv']
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        table.to_csv(output_csv, index=False)
        logging.info(f"Experiment completed. Results saved to {output_csv}")
        
    except Exception as e:
        logging.error(f"Fatal error during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
