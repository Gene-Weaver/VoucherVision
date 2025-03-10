import os, sys, inspect, json, shutil, cv2, time, glob #imagesize
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from tqdm import tqdm
from time import perf_counter
import concurrent.futures
from threading import Lock
from collections import defaultdict
import multiprocessing
import torch

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir = os.path.dirname(currentdir)
sys.path.append(currentdir)
from detect import run
sys.path.append(parentdir)
from landmark_processing import LeafSkeleton
from armature_processing import ArmatureSkeleton

def detect_plant_components(cfg, logger, dir_home, Project, Dirs):
    t1_start = perf_counter()
    logger.name = 'Locating Plant Components'
    logger.info(f"Detecting plant components in {len(os.listdir(Project.dir_images))} images")

    try:
        dir_exisiting_labels = cfg['leafmachine']['project']['use_existing_plant_component_detections']
    except:
        dir_exisiting_labels = None
    if cfg['leafmachine']['project']['num_workers'] is None:
        num_workers = 1
    else:
        num_workers = int(cfg['leafmachine']['project']['num_workers'])

    # Weights folder base
    dir_weights = os.path.join(dir_home, 'leafmachine2', 'component_detector','runs','train')
    
    # Detection threshold
    threshold = cfg['leafmachine']['plant_component_detector']['minimum_confidence_threshold']

    detector_version = cfg['leafmachine']['plant_component_detector']['detector_version']
    detector_iteration = cfg['leafmachine']['plant_component_detector']['detector_iteration']
    detector_weights = cfg['leafmachine']['plant_component_detector']['detector_weights']
    weights =  os.path.join(dir_weights,'Plant_Detector',detector_version,detector_iteration,'weights',detector_weights)

    do_save_prediction_overlay_images = not cfg['leafmachine']['plant_component_detector']['do_save_prediction_overlay_images']
    ignore_objects = cfg['leafmachine']['plant_component_detector']['ignore_objects_for_overlay']
    ignore_objects = ignore_objects or []

    if dir_exisiting_labels != None:
        logger.info("Loading existing plant labels")
        fetch_labels(dir_exisiting_labels, os.path.join(Dirs.path_plant_components, 'labels'))
        if len(Project.dir_images) <= 4000:
            logger.debug("Single-threaded create_dictionary_from_txt() len(Project.dir_images) <= 4000")
            A = create_dictionary_from_txt(logger, dir_exisiting_labels, 'Detections_Plant_Components', Project)
        else:
            logger.debug(f"Multi-threaded with ({str(cfg['leafmachine']['project']['num_workers'])}) threads create_dictionary_from_txt() len(Project.dir_images) > 4000")
            A = create_dictionary_from_txt_parallel(logger, cfg, dir_exisiting_labels, 'Detections_Plant_Components', Project)

    else:
        logger.info("Running YOLOv5 to generate plant labels")
        # run(weights = weights,
        #     source = Project.dir_images,
        #     project = Dirs.path_plant_components,
        #     name = Dirs.run_name,
        #     imgsz = (1280, 1280),
        #     nosave = do_save_prediction_overlay_images,
        #     anno_type = 'Plant_Detector',
        #     conf_thres = threshold, 
        #     ignore_objects_for_overlay = ignore_objects,
        #     mode = 'LM2',
        #     LOGGER=logger,)
        source = Project.dir_images
        project = Dirs.path_plant_components
        name = Dirs.run_name
        imgsz = (1280, 1280)
        nosave = do_save_prediction_overlay_images
        anno_type = 'Plant_Detector'
        conf_thres = threshold
        ignore_objects_for_overlay = ignore_objects
        mode = 'LM2'
        LOGGER = logger

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_in_parallel, weights, source, project, name, imgsz, nosave, anno_type,
                                    conf_thres, 10, ignore_objects_for_overlay, mode, LOGGER, i, num_workers) for i in
                    range(num_workers)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    _ = future.result()
                except Exception as e:
                    logger.error(f'Error in thread: {e}')
                    continue

        t2_stop = perf_counter()
        logger.info(f"[Plant components detection elapsed time] {round(t2_stop - t1_start)} seconds")
        logger.info(f"Threads [{num_workers}]")

        if len(Project.dir_images) <= 4000:
            logger.debug("Single-threaded create_dictionary_from_txt() len(Project.dir_images) <= 4000")
            A = create_dictionary_from_txt(logger, os.path.join(Dirs.path_plant_components, 'labels'), 'Detections_Plant_Components', Project)
        else:
            logger.debug(f"Multi-threaded with ({str(cfg['leafmachine']['project']['num_workers'])}) threads create_dictionary_from_txt() len(Project.dir_images) > 4000")
            A = create_dictionary_from_txt_parallel(logger, cfg, os.path.join(Dirs.path_plant_components, 'labels'), 'Detections_Plant_Components', Project)
    
    dict_to_json(Project.project_data, Dirs.path_plant_components, 'Detections_Plant_Components.json')
    
    t1_stop = perf_counter()
    logger.info(f"[Processing plant components elapsed time] {round(t1_stop - t1_start)} seconds")
    torch.cuda.empty_cache()
    return Project
    

def detect_archival_components(cfg, logger, dir_home, Project, Dirs, is_real_run=False, progress_report=None):
    print(f"cfg['leafmachine']['use_RGB_label_images'] {cfg['leafmachine']['use_RGB_label_images']}")
    if cfg['leafmachine']['use_RGB_label_images'] not in [1, '1']:
        logger.name = 'Skipping LeafMachine2 Label Detection'
        logger.info(f"Full image will be used instead of the label collage")  
        if is_real_run:
            progress_report.update_overall(f"Skipping LeafMachine2 Label Detection")             
    else:
        t1_start = perf_counter()
        logger.name = 'Locating Archival Components'
        logger.info(f"Detecting archival components in {len(os.listdir(Project.dir_images))} images")
        if is_real_run:
            progress_report.update_overall(f"Creating LeafMachine2 Label Collage") 

        
        try:
            dir_exisiting_labels = cfg['leafmachine']['project']['use_existing_archival_component_detections']
        except:
            dir_exisiting_labels = None
        if cfg['leafmachine']['project']['num_workers'] is None:
            num_workers = 1
        else:
            num_workers = int(cfg['leafmachine']['project']['num_workers'])

        min([len(os.listdir(Project.dir_images)), ])
        
        # Weights folder base
        dir_weights = os.path.join(dir_home, 'leafmachine2', 'component_detector','runs','train')
        
        # Detection threshold
        threshold = cfg['leafmachine']['archival_component_detector']['minimum_confidence_threshold']

        detector_version = cfg['leafmachine']['archival_component_detector']['detector_version']
        detector_iteration = cfg['leafmachine']['archival_component_detector']['detector_iteration']
        detector_weights = cfg['leafmachine']['archival_component_detector']['detector_weights']
        weights =  os.path.join(dir_weights,'Archival_Detector',detector_version,detector_iteration,'weights',detector_weights)

        do_save_prediction_overlay_images = not cfg['leafmachine']['archival_component_detector']['do_save_prediction_overlay_images']
        ignore_objects = cfg['leafmachine']['archival_component_detector']['ignore_objects_for_overlay']
        ignore_objects = ignore_objects or []


        if dir_exisiting_labels != None:
            logger.info("Loading existing archival labels")
            fetch_labels(dir_exisiting_labels, os.path.join(Dirs.path_archival_components, 'labels'))
            if len(Project.dir_images) <= 4000:
                logger.debug("Single-threaded create_dictionary_from_txt() len(Project.dir_images) <= 4000")
                A = create_dictionary_from_txt(logger, dir_exisiting_labels, 'Detections_Archival_Components', Project)
            else:
                logger.debug(f"Multi-threaded with ({str(cfg['leafmachine']['project']['num_workers'])}) threads create_dictionary_from_txt() len(Project.dir_images) > 4000")
                A = create_dictionary_from_txt_parallel(logger, cfg, dir_exisiting_labels, 'Detections_Archival_Components', Project)

        else:
            logger.info("Running YOLOv5 to generate archival labels")
            # run(weights = weights,
            #     source = Project.dir_images,
            #     project = Dirs.path_archival_components,
            #     name = Dirs.run_name,
            #     imgsz = (1280, 1280),
            #     nosave = do_save_prediction_overlay_images,
            #     anno_type = 'Archival_Detector',
            #     conf_thres = threshold, 
            #     ignore_objects_for_overlay = ignore_objects,
            #     mode = 'LM2',
            #     LOGGER=logger)
            # split the image paths into 4 chunks
            source = Project.dir_images
            project = Dirs.path_archival_components
            name = Dirs.run_name
            imgsz = (1280, 1280)
            nosave = do_save_prediction_overlay_images
            anno_type = 'Archival_Detector'
            conf_thres = threshold
            ignore_objects_for_overlay = ignore_objects
            mode = 'LM2'
            LOGGER = logger

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(run_in_parallel, weights, source, project, name, imgsz, nosave, anno_type,
                                        conf_thres, 10, ignore_objects_for_overlay, mode, LOGGER, i, num_workers) for i in
                        range(num_workers)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        _ = future.result()
                    except Exception as e:
                        logger.error(f'Error in thread: {e}')
                        continue

            t2_stop = perf_counter()
            logger.info(f"[Archival components detection elapsed time] {round(t2_stop - t1_start)} seconds")
            logger.info(f"Threads [{num_workers}]")

            if len(Project.dir_images) <= 4000:
                logger.debug("Single-threaded create_dictionary_from_txt() len(Project.dir_images) <= 4000")
                A = create_dictionary_from_txt(logger, os.path.join(Dirs.path_archival_components, 'labels'), 'Detections_Archival_Components', Project)
            else:
                logger.debug(f"Multi-threaded with ({str(cfg['leafmachine']['project']['num_workers'])}) threads create_dictionary_from_txt() len(Project.dir_images) > 4000")
                A = create_dictionary_from_txt_parallel(logger, cfg, os.path.join(Dirs.path_archival_components, 'labels'), 'Detections_Archival_Components', Project)
        
        dict_to_json(Project.project_data, Dirs.path_archival_components, 'Detections_Archival_Components.json')

        t1_stop = perf_counter()
        logger.info(f"[Processing archival components elapsed time] {round(t1_stop - t1_start)} seconds")
        torch.cuda.empty_cache()
    return Project


def detect_armature_components(cfg, logger, dir_home, Project, Dirs):
    t1_start = perf_counter()
    logger.name = 'Locating Armature Components'
    logger.info(f"Detecting armature components in {len(os.listdir(Project.dir_images))} images")

    if cfg['leafmachine']['project']['num_workers'] is None:
        num_workers = 1
    else:
        num_workers = int(cfg['leafmachine']['project']['num_workers'])

    # Weights folder base
    dir_weights = os.path.join(dir_home, 'leafmachine2', 'component_detector','runs','train')
    
    # Detection threshold
    threshold = cfg['leafmachine']['armature_component_detector']['minimum_confidence_threshold']

    detector_version = cfg['leafmachine']['armature_component_detector']['detector_version']
    detector_iteration = cfg['leafmachine']['armature_component_detector']['detector_iteration']
    detector_weights = cfg['leafmachine']['armature_component_detector']['detector_weights']
    weights =  os.path.join(dir_weights,'Armature_Detector',detector_version,detector_iteration,'weights',detector_weights)

    do_save_prediction_overlay_images = not cfg['leafmachine']['armature_component_detector']['do_save_prediction_overlay_images']
    ignore_objects = cfg['leafmachine']['armature_component_detector']['ignore_objects_for_overlay']
    ignore_objects = ignore_objects or []

    logger.info("Running YOLOv5 to generate armature labels")

    source = Project.dir_images
    project = Dirs.path_armature_components
    name = Dirs.run_name
    imgsz = (1280, 1280)
    nosave = do_save_prediction_overlay_images
    anno_type = 'Armature_Detector'
    conf_thres = threshold
    ignore_objects_for_overlay = ignore_objects
    mode = 'LM2'
    LOGGER = logger

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_in_parallel, weights, source, project, name, imgsz, nosave, anno_type,
                                conf_thres, 10, ignore_objects_for_overlay, mode, LOGGER, i, num_workers) for i in
                range(num_workers)]
        for future in concurrent.futures.as_completed(futures):
            try:
                _ = future.result()
            except Exception as e:
                logger.error(f'Error in thread: {e}')
                continue

    t2_stop = perf_counter()
    logger.info(f"[Plant components detection elapsed time] {round(t2_stop - t1_start)} seconds")
    logger.info(f"Threads [{num_workers}]")

    if len(Project.dir_images) <= 4000:
        logger.debug("Single-threaded create_dictionary_from_txt() len(Project.dir_images) <= 4000")
        A = create_dictionary_from_txt(logger, os.path.join(Dirs.path_armature_components, 'labels'), 'Detections_Armature_Components', Project)
    else:
        logger.debug(f"Multi-threaded with ({str(cfg['leafmachine']['project']['num_workers'])}) threads create_dictionary_from_txt() len(Project.dir_images) > 4000")
        A = create_dictionary_from_txt_parallel(logger, cfg, os.path.join(Dirs.path_armature_components, 'labels'), 'Detections_Armature_Components', Project)

    dict_to_json(Project.project_data, Dirs.path_armature_components, 'Detections_Armature_Components.json')
    
    t1_stop = perf_counter()
    logger.info(f"[Processing armature components elapsed time] {round(t1_stop - t1_start)} seconds")
    torch.cuda.empty_cache()
    return Project


''' RUN IN PARALLEL'''
def run_in_parallel(weights, source, project, name, imgsz, nosave, anno_type, conf_thres, line_thickness, ignore_objects_for_overlay, mode, LOGGER, chunk, n_workers):
    num_files = len(os.listdir(source))
    LOGGER.info(f"The number of worker threads: ({n_workers}), number of files ({num_files}).")

    chunk_size = len(os.listdir(source)) // n_workers
    start = chunk * chunk_size
    end = start + chunk_size if chunk < (n_workers-1) else len(os.listdir(source))

    sub_source = [os.path.join(source, f) for f in os.listdir(source)[start:end] if f.lower().endswith('.jpg')]

    run(weights=weights,
        source=sub_source,
        project=project,
        name=name,
        imgsz=imgsz,
        nosave=nosave,
        anno_type=anno_type,
        conf_thres=conf_thres,
        ignore_objects_for_overlay=ignore_objects_for_overlay,
        mode=mode,
        LOGGER=LOGGER)

''' RUN IN PARALLEL'''


###### Multi-thread NOTE this works, but unless there are several thousand images, it will be slower
def process_file(logger, file, dir_components, component, Project, lock):
    file_name = str(file.split('.')[0])
    with open(os.path.join(dir_components, file), "r") as f:
        with lock:
            Project.project_data[file_name][component] = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
            try:
                image_path = glob.glob(os.path.join(Project.dir_images, file_name + '.*'))[0]
                name_ext = os.path.basename(image_path)
                with Image.open(image_path) as im:
                    _, ext = os.path.splitext(name_ext)
                    if ext not in ['.jpg']:
                        im = im.convert('RGB')
                        im.save(os.path.join(Project.dir_images, file_name) + '.jpg', quality=100)
                        # file_name += '.jpg'
                    width, height = im.size
            except Exception as e:
                print(f"Unable to get image dimensions. Error: {e}")
                logger.info(f"Unable to get image dimensions. Error: {e}")
                width, height = None, None
            if width and height:
                Project.project_data[file_name]['height'] = int(height)
                Project.project_data[file_name]['width'] = int(width)


def create_dictionary_from_txt_parallel(logger, cfg, dir_components, component, Project):
    if cfg['leafmachine']['project']['num_workers'] is None:
        num_workers = 4 
    else:
        num_workers = int(cfg['leafmachine']['project']['num_workers'])

    files = [file for file in os.listdir(dir_components) if file.endswith(".txt")]
    lock = Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for file in files:
            futures.append(executor.submit(process_file, logger, file, dir_components, component, Project, lock))
        for future in concurrent.futures.as_completed(futures):
            pass
    return Project.project_data

######





# Single threaded
def create_dictionary_from_txt(logger, dir_components, component, Project):
    # dict_labels = {}
    for file in tqdm(os.listdir(dir_components), desc="Loading Annotations", colour='green'):
        if file.endswith(".txt"):
            file_name = str(file.split('.')[0])
            with open(os.path.join(dir_components, file), "r") as f:
                # dict_labels[file] = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                try:
                    Project.project_data[file_name][component] = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                except Exception as e:
                    print(e)
                    
                try:
                    image_path = glob.glob(os.path.join(Project.dir_images, file_name + '.*'))[0]
                    name_ext = os.path.basename(image_path)
                    with Image.open(image_path) as im:
                        _, ext = os.path.splitext(name_ext)
                        if ext not in ['.jpg']:
                            im = im.convert('RGB')
                            im.save(os.path.join(Project.dir_images, file_name) + '.jpg', quality=100)
                            # file_name += '.jpg'
                        width, height = im.size
                except Exception as e:
                    # print(f"Unable to get image dimensions. Error: {e}")
                    logger.info(f"Unable to get image dimensions. Error: {e}")
                    width, height = None, None
                if width and height:
                    Project.project_data[file_name]['height'] = int(height)
                    Project.project_data[file_name]['width'] = int(width)
    # for key, value in dict_labels.items():
    #     print(f'{key}  --> {value}')
    return Project.project_data


# old below   
'''def create_dictionary_from_txt(dir_components, component, Project):
    # dict_labels = {}
    for file in os.listdir(dir_components):
        if file.endswith(".txt"):
            file_name = str(file.split('.')[0])
            with open(os.path.join(dir_components, file), "r") as f:
                # dict_labels[file] = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                Project.project_data[file_name][component] = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                try:
                    width, height = imagesize.get(os.path.join(Project.dir_images, '.'.join([file_name,'jpg'])))
                except Exception as e:
                    print(f"Image not in 'jpg' format. Trying 'jpeg'. Note that other formats are not supported.{e}")
                    width, height = imagesize.get(os.path.join(Project.dir_images, '.'.join([file_name,'jpeg'])))
                Project.project_data[file_name]['height'] = int(height)
                Project.project_data[file_name]['width'] = int(width)
    # for key, value in dict_labels.items():
    #     print(f'{key}  --> {value}')
    return Project.project_data'''



def dict_to_json(dict_labels, dir_components, name_json):
    dir_components = os.path.join(dir_components, 'JSON')
    with open(os.path.join(dir_components, name_json), "w") as outfile:
        json.dump(dict_labels, outfile,sort_keys=False)

def fetch_labels(dir_exisiting_labels, new_dir):
    shutil.copytree(dir_exisiting_labels, new_dir)


'''Landmarks - uses YOLO, but works differently than above. A hybrid between segmentation and component detector'''
def detect_landmarks(cfg, logger, dir_home, Project, batch, n_batches, Dirs, segmentation_complete):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Detect Landmarks]'
    logger.info(f'Detecting landmarks for batch {batch+1} of {n_batches}')

    landmark_whole_leaves = cfg['leafmachine']['landmark_detector']['landmark_whole_leaves']
    landmark_partial_leaves = cfg['leafmachine']['landmark_detector']['landmark_partial_leaves']

    landmarks_whole_leaves_props = {}
    landmarks_whole_leaves_overlay = {}
    landmarks_partial_leaves_props = {}
    landmarks_partial_leaves_overlay = {}

    if landmark_whole_leaves:
        run_landmarks(cfg, logger, dir_home, Project, batch, n_batches, Dirs, 'Landmarks_Whole_Leaves', segmentation_complete)
    if landmark_partial_leaves:
        run_landmarks(cfg, logger, dir_home, Project, batch, n_batches, Dirs, 'Landmarks_Partial_Leaves', segmentation_complete)

    # if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
    #     landmarks_whole_leaves_props_batch, landmarks_whole_leaves_overlay_batch = run_landmarks(Instance_Detector_Whole, Project.project_data_list[batch], 0, 
    #                                                                                 "Segmentation_Whole_Leaf", "Whole_Leaf_Cropped", cfg, Project, Dirs, batch, n_batches)#, start+1, end)
    #     landmarks_whole_leaves_props.update(landmarks_whole_leaves_props_batch)
    #     landmarks_whole_leaves_overlay.update(landmarks_whole_leaves_overlay_batch)
    # if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
    #     landmarks_partial_leaves_props_batch, landmarks_partial_leaves_overlay_batch = run_landmarks(Instance_Detector_Partial, Project.project_data_list[batch], 1, 
    #                                                                                 "Segmentation_Partial_Leaf", "Partial_Leaf_Cropped", cfg, Project, Dirs, batch, n_batches)#, start+1, end)
    #     landmarks_partial_leaves_props.update(landmarks_partial_leaves_props_batch)
    #     landmarks_partial_leaves_overlay.update(landmarks_partial_leaves_overlay_batch)
    
    end_t = perf_counter()
    logger.info(f'Batch {batch+1}/{n_batches}: Landmark Detection Duration --> {round((end_t - start_t)/60)} minutes')
    return Project


def detect_armature(cfg, logger, dir_home, Project, batch, n_batches, Dirs, segmentation_complete):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Detect Armature]'
    logger.info(f'Detecting armature for batch {batch+1} of {n_batches}')

    landmark_armature = cfg['leafmachine']['modules']['armature']

    landmarks_armature_props = {}
    landmarks_armature_overlay = {}

    if landmark_armature:
        run_armature(cfg, logger, dir_home, Project, batch, n_batches, Dirs, 'Landmarks_Armature', segmentation_complete)

    end_t = perf_counter()
    logger.info(f'Batch {batch+1}/{n_batches}: Armature Detection Duration --> {round((end_t - start_t)/60)} minutes')
    return Project


def run_armature(cfg, logger, dir_home, Project, batch, n_batches, Dirs, leaf_type, segmentation_complete):
    
    logger.info('Detecting armature landmarks from scratch')
    if leaf_type == 'Landmarks_Armature':
        dir_overlay = os.path.join(Dirs.landmarks_armature_overlay, ''.join(['batch_',str(batch+1)]))

    # if not segmentation_complete: # If segmentation was run, then don't redo the unpack, just do the crop into the temp folder
    if leaf_type == 'Landmarks_Armature': # TODO THE 0 is for prickles. For spines I'll need to add a 1 like with partial_leaves or just do it for all
        Project.project_data_list[batch] = unpack_class_from_components_armature(Project.project_data_list[batch], 0, 'Armature_YOLO', 'Armature_BBoxes', Project)
        Project.project_data_list[batch], dir_temp = crop_images_to_bbox_armature(Project.project_data_list[batch], 0, 'Armature_Cropped', "Armature_BBoxes", Project, Dirs, True, cfg)

    # Weights folder base
    dir_weights = os.path.join(dir_home, 'leafmachine2', 'component_detector','runs','train')
    
    # Detection threshold
    threshold = cfg['leafmachine']['landmark_detector_armature']['minimum_confidence_threshold']

    detector_version = cfg['leafmachine']['landmark_detector_armature']['detector_version']
    detector_iteration = cfg['leafmachine']['landmark_detector_armature']['detector_iteration']
    detector_weights = cfg['leafmachine']['landmark_detector_armature']['detector_weights']
    weights =  os.path.join(dir_weights,'Landmark_Detector_YOLO',detector_version,detector_iteration,'weights',detector_weights)

    do_save_prediction_overlay_images = not cfg['leafmachine']['landmark_detector_armature']['do_save_prediction_overlay_images']
    ignore_objects = cfg['leafmachine']['landmark_detector_armature']['ignore_objects_for_overlay']
    ignore_objects = ignore_objects or []
    if cfg['leafmachine']['project']['num_workers'] is None:
        num_workers = 1
    else:
        num_workers = int(cfg['leafmachine']['project']['num_workers'])

    has_images = False
    if len(os.listdir(dir_temp)) > 0:
        has_images = True
        source = dir_temp
        project = dir_overlay
        name = Dirs.run_name
        imgsz = (1280, 1280)
        nosave = do_save_prediction_overlay_images
        anno_type = 'Armature_Detector'
        conf_thres = threshold
        line_thickness = 2
        ignore_objects_for_overlay = ignore_objects
        mode = 'Landmark'
        LOGGER = logger

        # Initialize a Lock object to ensure thread safety
        lock = Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_in_parallel, weights, source, project, name, imgsz, nosave, anno_type,
                                    conf_thres, line_thickness, ignore_objects_for_overlay, mode, LOGGER, i, num_workers) for i in
                    range(num_workers)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    _ = future.result()
                except Exception as e:
                    logger.error(f'Error in thread: {e}')
                    continue

        with lock:
            if has_images:
                dimensions_dict = get_cropped_dimensions(dir_temp)
                A = add_to_dictionary_from_txt_armature(cfg, logger, Dirs, leaf_type, os.path.join(dir_overlay, 'labels'), leaf_type, Project, dimensions_dict, dir_temp, batch, n_batches)
            else:
                # TODO add empty placeholder to the image data
                pass
    
    # delete the temp dir
    try:
        shutil.rmtree(dir_temp)
    except:
        try:
            time.sleep(5)
            shutil.rmtree(dir_temp)
        except:
            try:
                time.sleep(5)
                shutil.rmtree(dir_temp)
            except:
                pass

    torch.cuda.empty_cache()

    return Project


def run_landmarks(cfg, logger, dir_home, Project, batch, n_batches, Dirs, leaf_type, segmentation_complete):
    use_existing_landmark_detections = cfg['leafmachine']['landmark_detector']['use_existing_landmark_detections']
    
    if use_existing_landmark_detections is None:
        logger.info('Detecting landmarks from scratch')
        if leaf_type == 'Landmarks_Whole_Leaves':
            dir_overlay = os.path.join(Dirs.landmarks_whole_leaves_overlay, ''.join(['batch_',str(batch+1)]))
        elif leaf_type == 'Landmarks_Partial_Leaves':
            dir_overlay = os.path.join(Dirs.landmarks_partial_leaves_overlay, ''.join(['batch_',str(batch+1)]))

        # if not segmentation_complete: # If segmentation was run, then don't redo the unpack, just do the crop into the temp folder
        if leaf_type == 'Landmarks_Whole_Leaves':
            Project.project_data_list[batch] = unpack_class_from_components(Project.project_data_list[batch], 0, 'Whole_Leaf_BBoxes_YOLO', 'Whole_Leaf_BBoxes', Project)
            Project.project_data_list[batch], dir_temp = crop_images_to_bbox(Project.project_data_list[batch], 0, 'Whole_Leaf_Cropped', "Whole_Leaf_BBoxes", Project, Dirs)

        elif leaf_type == 'Landmarks_Partial_Leaves':
            Project.project_data_list[batch] = unpack_class_from_components(Project.project_data_list[batch], 1, 'Partial_Leaf_BBoxes_YOLO', 'Partial_Leaf_BBoxes', Project)
            Project.project_data_list[batch], dir_temp = crop_images_to_bbox(Project.project_data_list[batch], 1, 'Partial_Leaf_Cropped', "Partial_Leaf_BBoxes", Project, Dirs)
        # else:
        #     if leaf_type == 'Landmarks_Whole_Leaves':
        #         Project.project_data_list[batch], dir_temp = crop_images_to_bbox(Project.project_data_list[batch], 0, 'Whole_Leaf_Cropped', "Whole_Leaf_BBoxes", Project, Dirs)
        #     elif leaf_type == 'Landmarks_Partial_Leaves':
        #         Project.project_data_list[batch], dir_temp = crop_images_to_bbox(Project.project_data_list[batch], 1, 'Partial_Leaf_Cropped', "Partial_Leaf_BBoxes", Project, Dirs)

        # Weights folder base
        dir_weights = os.path.join(dir_home, 'leafmachine2', 'component_detector','runs','train')
        
        # Detection threshold
        threshold = cfg['leafmachine']['landmark_detector']['minimum_confidence_threshold']

        detector_version = cfg['leafmachine']['landmark_detector']['detector_version']
        detector_iteration = cfg['leafmachine']['landmark_detector']['detector_iteration']
        detector_weights = cfg['leafmachine']['landmark_detector']['detector_weights']
        weights =  os.path.join(dir_weights,'Landmark_Detector_YOLO',detector_version,detector_iteration,'weights',detector_weights)

        do_save_prediction_overlay_images = not cfg['leafmachine']['landmark_detector']['do_save_prediction_overlay_images']
        ignore_objects = cfg['leafmachine']['landmark_detector']['ignore_objects_for_overlay']
        ignore_objects = ignore_objects or []
        if cfg['leafmachine']['project']['num_workers'] is None:
            num_workers = 1
        else:
            num_workers = int(cfg['leafmachine']['project']['num_workers'])

        has_images = False
        if len(os.listdir(dir_temp)) > 0:
            has_images = True
            # run(weights = weights,
            #     source = dir_temp,
            #     project = dir_overlay,
            #     name = Dirs.run_name,
            #     imgsz = (1280, 1280),
            #     nosave = do_save_prediction_overlay_images,
            #     anno_type = 'Landmark_Detector_YOLO',
            #     conf_thres = threshold, 
            #     line_thickness = 2,
            #     ignore_objects_for_overlay = ignore_objects,
            #     mode = 'Landmark')
            source = dir_temp
            project = dir_overlay
            name = Dirs.run_name
            imgsz = (1280, 1280)
            nosave = do_save_prediction_overlay_images
            anno_type = 'Landmark_Detector'
            conf_thres = threshold
            line_thickness = 2
            ignore_objects_for_overlay = ignore_objects
            mode = 'Landmark'
            LOGGER = logger

            # Initialize a Lock object to ensure thread safety
            lock = Lock()

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(run_in_parallel, weights, source, project, name, imgsz, nosave, anno_type,
                                        conf_thres, line_thickness, ignore_objects_for_overlay, mode, LOGGER, i, num_workers) for i in
                        range(num_workers)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        _ = future.result()
                    except Exception as e:
                        logger.error(f'Error in thread: {e}')
                        continue

            with lock:
                if has_images:
                    dimensions_dict = get_cropped_dimensions(dir_temp)
                    A = add_to_dictionary_from_txt(cfg, logger, Dirs, leaf_type, os.path.join(dir_overlay, 'labels'), leaf_type, Project, dimensions_dict, dir_temp, batch, n_batches)
                else:
                    # TODO add empty placeholder to the image data
                    pass
    else:
        logger.info('Loading existing landmark annotations')
        dir_temp = os.path.join(use_existing_landmark_detections, f'batch_{str(batch+1)}', 'labels')
        dimensions_dict = get_cropped_dimensions(dir_temp)
        A = add_to_dictionary_from_txt(cfg, logger, Dirs, leaf_type, use_existing_landmark_detections, leaf_type, Project, dimensions_dict, dir_temp, batch, n_batches)

    
    # delete the temp dir
    try:
        shutil.rmtree(dir_temp)
    except:
        try:
            time.sleep(5)
            shutil.rmtree(dir_temp)
        except:
            try:
                time.sleep(5)
                shutil.rmtree(dir_temp)
            except:
                pass

    torch.cuda.empty_cache()

    return Project
    '''def add_to_dictionary_from_txt(cfg, Dirs, leaf_type, dir_components, component, Project, dimensions_dict, dir_temp):
    # dict_labels = {}
    for file in os.listdir(dir_components):
        file_name = str(file.split('.')[0])
        file_name_parent = file_name.split('__')[0]
        Project.project_data[file_name_parent][component] = {}

        if file.endswith(".txt"):
            with open(os.path.join(dir_components, file), "r") as f:
                all_points = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                Project.project_data[file_name_parent][component][file_name] = all_points
                
                height = dimensions_dict[file_name][0]
                width = dimensions_dict[file_name][1]

                Leaf_Skeleton = LeafSkeleton(cfg, Dirs, leaf_type, all_points, height, width, dir_temp, file_name)
                QC_add = Leaf_Skeleton.get_QC()'''


    return Project.project_data

def add_to_dictionary_from_txt_armature(cfg, logger, Dirs, leaf_type, dir_components, component, Project, dimensions_dict, dir_temp, batch, n_batches):
    dpi = cfg['leafmachine']['overlay']['overlay_dpi']
    if leaf_type == 'Landmarks_Armature':
        logger.info(f'Detecting landmarks armature')
        pdf_path = os.path.join(Dirs.landmarks_armature_overlay_QC, ''.join(['landmarks_armature_overlay_QC__',str(batch+1), 'of', str(n_batches), '.pdf']))
        pdf_path_final = os.path.join(Dirs.landmarks_armature_overlay_final, ''.join(['landmarks_armature_overlay_final__',str(batch+1), 'of', str(n_batches), '.pdf']))

    ### FINAL
    # dict_labels = {}
    fig = plt.figure(figsize=(8.27, 11.69), dpi=dpi) # A4 size, 300 dpi
    row, col = 0, 0
    with PdfPages(pdf_path_final) as pdf:
        
        

        for file in os.listdir(dir_components):
            file_name = str(file.split('.')[0])
            file_name_parent = file_name.split('__')[0]

            # Project.project_data_list[batch][file_name_parent][component] = []

            if file_name_parent in Project.project_data_list[batch]:

                

                if file.endswith(".txt"):
                    with open(os.path.join(dir_components, file), "r") as f:
                        all_points = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                        # Project.project_data_list[batch][file_name_parent][component][file_name] = all_points

                        height = dimensions_dict[file_name][0]
                        width = dimensions_dict[file_name][1]

                        Armature_Skeleton = ArmatureSkeleton(cfg, logger, Dirs, leaf_type, all_points, height, width, dir_temp, file_name)
                        Project = add_armature_skeleton_to_project(cfg, logger, Project, batch, file_name_parent, component, Dirs, leaf_type, all_points, height, width, dir_temp, file_name, Armature_Skeleton)
                        final_add = cv2.cvtColor(Armature_Skeleton.get_final(), cv2.COLOR_BGR2RGB)

                        # Add image to the current subplot
                        ax = fig.add_subplot(5, 3, row * 3 + col + 1)
                        ax.imshow(final_add)
                        ax.axis('off')

                        col += 1
                        if col == 3:
                            col = 0
                            row += 1
                        if row == 5:
                            row = 0
                            pdf.savefig(fig)  # Save the current page
                            fig = plt.figure(figsize=(8.27, 11.69), dpi=300) # Create a new page
            else:
                pass

        if row != 0 or col != 0:
            pdf.savefig(fig)  # Save the remaining images on the last page

def add_to_dictionary_from_txt(cfg, logger, Dirs, leaf_type, dir_components, component, Project, dimensions_dict, dir_temp, batch, n_batches):
    dpi = cfg['leafmachine']['overlay']['overlay_dpi']
    if leaf_type == 'Landmarks_Whole_Leaves':
        logger.info(f'Detecting landmarks whole leaves')
        pdf_path = os.path.join(Dirs.landmarks_whole_leaves_overlay_QC, ''.join(['landmarks_whole_leaves_overlay_QC__',str(batch+1), 'of', str(n_batches), '.pdf']))
        pdf_path_final = os.path.join(Dirs.landmarks_whole_leaves_overlay_final, ''.join(['landmarks_whole_leaves_overlay_final__',str(batch+1), 'of', str(n_batches), '.pdf']))
    elif leaf_type == 'Landmarks_Partial_Leaves':
        logger.info(f'Detecting landmarks partial leaves')
        pdf_path = os.path.join(Dirs.landmarks_partial_leaves_overlay_QC, ''.join(['landmarks_partial_leaves_overlay_QC__',str(batch+1), 'of', str(n_batches), '.pdf']))
        pdf_path_final = os.path.join(Dirs.landmarks_partial_leaves_overlay_final, ''.join(['landmarks_partial_leaves_overlay_final__',str(batch+1), 'of', str(n_batches), '.pdf']))
    elif leaf_type == 'Landmarks_Armature':
        logger.info(f'Detecting landmarks armature')
        pdf_path = os.path.join(Dirs.landmarks_armature_overlay_QC, ''.join(['landmarks_armature_overlay_QC__',str(batch+1), 'of', str(n_batches), '.pdf']))
        pdf_path_final = os.path.join(Dirs.landmarks_armature_overlay_final, ''.join(['landmarks_armature_overlay_final__',str(batch+1), 'of', str(n_batches), '.pdf']))

    ### FINAL
    # dict_labels = {}
    fig = plt.figure(figsize=(8.27, 11.69), dpi=dpi) # A4 size, 300 dpi
    row, col = 0, 0
    with PdfPages(pdf_path_final) as pdf:
        
        

        for file in os.listdir(dir_components):
            file_name = str(file.split('.')[0])
            file_name_parent = file_name.split('__')[0]

            # Project.project_data_list[batch][file_name_parent][component] = []

            if file_name_parent in Project.project_data_list[batch]:

                

                if file.endswith(".txt"):
                    with open(os.path.join(dir_components, file), "r") as f:
                        all_points = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                        # Project.project_data_list[batch][file_name_parent][component][file_name] = all_points

                        height = dimensions_dict[file_name][0]
                        width = dimensions_dict[file_name][1]

                        Leaf_Skeleton = LeafSkeleton(cfg, logger, Dirs, leaf_type, all_points, height, width, dir_temp, file_name)
                        Project = add_leaf_skeleton_to_project(cfg, logger, Project, batch, file_name_parent, component, Dirs, leaf_type, all_points, height, width, dir_temp, file_name, Leaf_Skeleton)
                        final_add = cv2.cvtColor(Leaf_Skeleton.get_final(), cv2.COLOR_BGR2RGB)

                        # Add image to the current subplot
                        ax = fig.add_subplot(5, 3, row * 3 + col + 1)
                        ax.imshow(final_add)
                        ax.axis('off')

                        col += 1
                        if col == 3:
                            col = 0
                            row += 1
                        if row == 5:
                            row = 0
                            pdf.savefig(fig)  # Save the current page
                            fig = plt.figure(figsize=(8.27, 11.69), dpi=300) # Create a new page
            else:
                pass

        if row != 0 or col != 0:
            pdf.savefig(fig)  # Save the remaining images on the last page

    ### QC
    '''do_save_QC_pdf = False # TODO refine this
    if do_save_QC_pdf:
        # dict_labels = {}
        fig = plt.figure(figsize=(8.27, 11.69), dpi=dpi) # A4 size, 300 dpi
        row, col = 0, 0
        with PdfPages(pdf_path) as pdf:



            for file in os.listdir(dir_components):
                file_name = str(file.split('.')[0])
                file_name_parent = file_name.split('__')[0]

                if file_name_parent in Project.project_data_list[batch]:

                    if file.endswith(".txt"):
                        with open(os.path.join(dir_components, file), "r") as f:
                            all_points = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                            Project.project_data_list[batch][file_name_parent][component][file_name] = all_points

                            height = dimensions_dict[file_name][0]
                            width = dimensions_dict[file_name][1]

                            Leaf_Skeleton = LeafSkeleton(cfg, logger, Dirs, leaf_type, all_points, height, width, dir_temp, file_name)
                            QC_add = cv2.cvtColor(Leaf_Skeleton.get_QC(), cv2.COLOR_BGR2RGB)

                            # Add image to the current subplot
                            ax = fig.add_subplot(5, 3, row * 3 + col + 1)
                            ax.imshow(QC_add)
                            ax.axis('off')

                            col += 1
                            if col == 3:
                                col = 0
                                row += 1
                            if row == 5:
                                row = 0
                                pdf.savefig(fig)  # Save the current page
                                fig = plt.figure(figsize=(8.27, 11.69), dpi=300) # Create a new page
                else:
                    pass

            if row != 0 or col != 0:
                pdf.savefig(fig)  # Save the remaining images on the last page'''


def add_armature_skeleton_to_project(cfg, logger, Project, batch, file_name_parent, component, Dirs, leaf_type, all_points, height, width, dir_temp, file_name, ARM):
    if ARM.is_complete:
        try:
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'armature_status': 'complete'}, {'armature': ARM}]})
        except:
            Project.project_data_list[batch][file_name_parent][component] = []
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'armature_status': 'complete'}, {'armature': ARM}]})

    else:
        try:
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'armature_status': 'incomplete'}, {'armature': ARM}]})
        except:
            Project.project_data_list[batch][file_name_parent][component] = []
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'armature_status': 'incomplete'}, {'armature': ARM}]})


    return Project


def add_leaf_skeleton_to_project(cfg, logger, Project, batch, file_name_parent, component, Dirs, leaf_type, all_points, height, width, dir_temp, file_name, LS):

    if LS.is_complete_leaf:
        try:
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'landmark_status': 'complete_leaf'}, {'landmarks': LS}]})
        except:
            Project.project_data_list[batch][file_name_parent][component] = []
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'landmark_status': 'complete_leaf'}, {'landmarks': LS}]})
        # Project.project_data_list[batch][file_name_parent][component][file_name].update({'landmark_status': 'complete_leaf'})
        # Project.project_data_list[batch][file_name_parent][component][file_name].update({'landmarks': LS})

    elif LS.is_leaf_no_width:
        try:
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'landmark_status': 'leaf_no_width'}, {'landmarks': LS}]})
        except:
            Project.project_data_list[batch][file_name_parent][component] = []
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'landmark_status': 'leaf_no_width'}, {'landmarks': LS}]})
        # Project.project_data_list[batch][file_name_parent][component][file_name].update({'landmark_status': 'leaf_no_width'})
        # Project.project_data_list[batch][file_name_parent][component][file_name].update({'landmarks': LS})

    else:
        try:
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'landmark_status': 'incomplete'}, {'landmarks': LS}]})
        except:
            Project.project_data_list[batch][file_name_parent][component] = []
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'landmark_status': 'incomplete'}, {'landmarks': LS}]})

        # Project.project_data_list[batch][file_name_parent][component][file_name].update({'landmark_status': 'incomplete'})
        # Project.project_data_list[batch][file_name_parent][component][file_name].update({'landmarks': LS})

    return Project


'''
self.determine_lamina_length('final') 

# Lamina tip and base
if self.has_lamina_tip:
    cv2.circle(self.image_final, self.lamina_tip, radius=4, color=(0, 255, 0), thickness=2)
    cv2.circle(self.image_final, self.lamina_tip, radius=2, color=(255, 255, 255), thickness=-1)
if self.has_lamina_base:
    cv2.circle(self.image_final, self.lamina_base, radius=4, color=(255, 0, 0), thickness=2)
    cv2.circle(self.image_final, self.lamina_base, radius=2, color=(255, 255, 255), thickness=-1)

# Apex angle
# if self.apex_center != []:
#     cv2.circle(self.image_final, self.apex_center, radius=3, color=(0, 255, 0), thickness=-1)
if self.apex_left != []:
    cv2.circle(self.image_final, self.apex_left, radius=3, color=(255, 0, 0), thickness=-1)
if self.apex_right != []:
    cv2.circle(self.image_final, self.apex_right, radius=3, color=(0, 0, 255), thickness=-1)

# Base angle
# if self.base_center:
#     cv2.circle(self.image_final, self.base_center, radius=3, color=(0, 255, 0), thickness=-1)
if self.base_left:
    cv2.circle(self.image_final, self.base_left, radius=3, color=(255, 0, 0), thickness=-1)
if self.base_right:
    cv2.circle(self.image_final, self.base_right, radius=3, color=(0, 0, 255), thickness=-1)

# Draw line of fit
for point in self.width_infer:


'''









def get_cropped_dimensions(dir_temp):
    dimensions_dict = {}
    for file_name in os.listdir(dir_temp):
        if file_name.endswith(".jpg"):
            img = cv2.imread(os.path.join(dir_temp, file_name))
            height, width, channels = img.shape
            stem = os.path.splitext(file_name)[0]
            dimensions_dict[stem] = (height, width)
    return dimensions_dict

def unpack_class_from_components_armature(dict_big, cls, dict_name_yolo, dict_name_location, Project):
    # Get the dict that contains plant parts, find the whole leaves
    for filename, value in dict_big.items():
        if "Detections_Armature_Components" in value:
            filtered_components = [val for val in value["Detections_Armature_Components"] if val[0] == cls]
            value[dict_name_yolo] = filtered_components

    for filename, value in dict_big.items():
        if "Detections_Armature_Components" in value:
            filtered_components = [val for val in value["Detections_Armature_Components"] if val[0] == cls]
            height = value['height']
            width = value['width']
            converted_list = [[convert_index_to_class_armature(val[0]), int((val[1] * width) - ((val[3] * width) / 2)), 
                                                                int((val[2] * height) - ((val[4] * height) / 2)), 
                                                                int(val[3] * width) + int((val[1] * width) - ((val[3] * width) / 2)), 
                                                                int(val[4] * height) + int((val[2] * height) - ((val[4] * height) / 2))] for val in filtered_components]
            # Verify that the crops are correct
            # img = Image.open(os.path.join(Project., '.'.join([filename,'jpg'])))
            # for d in converted_list:
            #     img_crop = img.crop((d[1], d[2], d[3], d[4]))
            #     img_crop.show() 
            value[dict_name_location] = converted_list
    # print(dict)
    return dict_big

def unpack_class_from_components(dict_big, cls, dict_name_yolo, dict_name_location, Project):
    # Get the dict that contains plant parts, find the whole leaves
    for filename, value in dict_big.items():
        if "Detections_Plant_Components" in value:
            filtered_components = [val for val in value["Detections_Plant_Components"] if val[0] == cls]
            value[dict_name_yolo] = filtered_components

    for filename, value in dict_big.items():
        if "Detections_Plant_Components" in value:
            filtered_components = [val for val in value["Detections_Plant_Components"] if val[0] == cls]
            height = value['height']
            width = value['width']
            converted_list = [[convert_index_to_class(val[0]), int((val[1] * width) - ((val[3] * width) / 2)), 
                                                                int((val[2] * height) - ((val[4] * height) / 2)), 
                                                                int(val[3] * width) + int((val[1] * width) - ((val[3] * width) / 2)), 
                                                                int(val[4] * height) + int((val[2] * height) - ((val[4] * height) / 2))] for val in filtered_components]
            # Verify that the crops are correct
            # img = Image.open(os.path.join(Project., '.'.join([filename,'jpg'])))
            # for d in converted_list:
            #     img_crop = img.crop((d[1], d[2], d[3], d[4]))
            #     img_crop.show() 
            value[dict_name_location] = converted_list
    # print(dict)
    return dict_big


def crop_images_to_bbox_armature(dict_big, cls, dict_name_cropped, dict_from, Project, Dirs, do_upscale=False, cfg=None):
    dir_temp = os.path.join(Dirs.landmarks, 'TEMP_landmarks')
    os.makedirs(dir_temp, exist_ok=True)
    # For each image, iterate through the whole leaves, segment, report data back to dict_plant_components
    for filename, value in dict_big.items():
        value[dict_name_cropped] = []
        if dict_from in value:
            bboxes_whole_leaves = [val for val in value[dict_from] if val[0] == convert_index_to_class_armature(cls)]
            if len(bboxes_whole_leaves) == 0:
                m = str(''.join(['No objects for class ', convert_index_to_class_armature(0), ' were found']))
                # Print_Verbose(cfg, 3, m).plain()
            else:
                try:
                    img = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpg'])))
                    # img = cv2.imread(os.path.join(Project, '.'.join([filename,'jpg']))) # Testing
                except:
                    img = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpeg'])))
                    # img = cv2.imread(os.path.join(Project, '.'.join([filename,'jpeg']))) # Testing
                
                for d in bboxes_whole_leaves:
                    # img_crop = img.crop((d[1], d[2], d[3], d[4])) # PIL
                    img_crop = img[d[2]:d[4], d[1]:d[3]]
                    loc = '-'.join([str(d[1]), str(d[2]), str(d[3]), str(d[4])])
                    # value[dict_name_cropped].append({crop_name: img_crop})
                    if do_upscale:
                        upscale_factor = int(cfg['leafmachine']['landmark_detector_armature']['upscale_factor'])
                        if cls == 0:
                            crop_name = '__'.join([filename,f"PRICKLE-{upscale_factor}x",loc])
                        height, width, _ = img_crop.shape
                        img_crop = cv2.resize(img_crop, ((width * upscale_factor), (height * upscale_factor)), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        if cls == 0:
                            crop_name = '__'.join([filename,'PRICKLE',loc])

                    cv2.imwrite(os.path.join(dir_temp, '.'.join([crop_name,'jpg'])), img_crop)
                    # cv2.imshow('img_crop', img_crop)
                    # cv2.waitKey(0)
                    # img_crop.show() # PIL
    return dict_big, dir_temp


def crop_images_to_bbox(dict_big, cls, dict_name_cropped, dict_from, Project, Dirs):
    dir_temp = os.path.join(Dirs.landmarks, 'TEMP_landmarks')
    os.makedirs(dir_temp, exist_ok=True)
    # For each image, iterate through the whole leaves, segment, report data back to dict_plant_components
    for filename, value in dict_big.items():
        value[dict_name_cropped] = []
        if dict_from in value:
            bboxes_whole_leaves = [val for val in value[dict_from] if val[0] == convert_index_to_class(cls)]
            if len(bboxes_whole_leaves) == 0:
                m = str(''.join(['No objects for class ', convert_index_to_class(0), ' were found']))
                # Print_Verbose(cfg, 3, m).plain()
            else:
                try:
                    img = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpg'])))
                    # img = cv2.imread(os.path.join(Project, '.'.join([filename,'jpg']))) # Testing
                except:
                    img = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpeg'])))
                    # img = cv2.imread(os.path.join(Project, '.'.join([filename,'jpeg']))) # Testing
                
                for d in bboxes_whole_leaves:
                    # img_crop = img.crop((d[1], d[2], d[3], d[4])) # PIL
                    img_crop = img[d[2]:d[4], d[1]:d[3]]
                    loc = '-'.join([str(d[1]), str(d[2]), str(d[3]), str(d[4])])
                    if cls == 0:
                        crop_name = '__'.join([filename,'L',loc])
                    elif cls == 1:
                        crop_name = '__'.join([filename,'PL',loc])
                    elif cls == 2:
                        crop_name = '__'.join([filename,'ARM',loc])
                    # value[dict_name_cropped].append({crop_name: img_crop})
                    cv2.imwrite(os.path.join(dir_temp, '.'.join([crop_name,'jpg'])), img_crop)
                    # cv2.imshow('img_crop', img_crop)
                    # cv2.waitKey(0)
                    # img_crop.show() # PIL
    return dict_big, dir_temp

def convert_index_to_class(ind):
    mapping = {
        0: 'apex_angle',
        1: 'base_angle',
        2: 'lamina_base',
        3: 'lamina_tip',
        4: 'lamina_width',
        5: 'lobe_tip',
        6: 'midvein_trace',
        7: 'petiole_tip',
        8: 'petiole_trace',
    }
    return mapping.get(ind, 'Invalid class').lower()

def convert_index_to_class_armature(ind):
    mapping = {
        0: 'tip',
        1: 'middle',
        2: 'outer',
    }
    return mapping.get(ind, 'Invalid class').lower()
