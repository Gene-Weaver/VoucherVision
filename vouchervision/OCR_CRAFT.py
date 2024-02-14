# import Craft class
from craft_text_detector import read_image, load_craftnet_model, load_refinenet_model, get_prediction, export_detected_regions, export_extra_results, empty_cuda_cache

def main2():
    # import craft functions
    

    # set image path and export folder directory
    # image = 'D:/Dropbox/SLTP/benchmark_datasets/SLTP_B50_MICH_Angiospermae2/img/MICH_7375774_Polygonaceae_Persicaria_.jpg' # can be filepath, PIL image or numpy array
    # image = 'C:/Users/Will/Downloads/test_2024_02_07__14-59-52/Original_Images/SJRw 00891 - 01141__10001.jpg'
    image = 'D:/Dropbox/VoucherVision/demo/demo_images/MICH_16205594_Poaceae_Jouvea_pilosa.jpg'
    output_dir = 'D:/D_Desktop/test_out_CRAFT'

    # read image
    image = read_image(image)

    # load models
    refine_net = load_refinenet_model(cuda=True)
    craft_net = load_craftnet_model(weight_path='D:/Dropbox/VoucherVision/vouchervision/craft/craft_mlt_25k.pth', cuda=True)

    # perform prediction
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.4,
        link_threshold=0.7,
        low_text=0.4,
        cuda=True,
        long_size=1280
    )

    # export detected text regions
    exported_file_paths = export_detected_regions(
        image=image,
        regions=prediction_result["boxes"],
        output_dir=output_dir,
        rectify=True
    )

    # export heatmap, detection points, box visualization
    export_extra_results(
        image=image,
        regions=prediction_result["boxes"],
        heatmaps=prediction_result["heatmaps"],
        output_dir=output_dir
    )

    # unload models from gpu
    empty_cuda_cache()


if __name__ == '__main__':
    # main()
    main2()