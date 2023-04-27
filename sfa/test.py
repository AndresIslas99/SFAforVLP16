"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script

114 esta el modelo 
136 estan los outputs
"""

import argparse
import sys
import os
import time
import warnings

import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import tf.transformations as tf_transformations

import threading
import tf2_ros
from geometry_msgs.msg import TransformStamped


warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.torch_utils import _sigmoid
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration


def publish_transform(x, y, z, qx, qy, qz, qw, reference_frame, marker_frame):
    broadcaster = tf2_ros.TransformBroadcaster()

    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = reference_frame
        transform.child_frame_id = marker_frame

        # Set the translation and rotation values
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw

        broadcaster.sendTransform(transform)

        rate.sleep()

def create_marker_array(detections):
    marker_array = MarkerArray()
    
    marker_id = 0
              
    
    for obj_class, obj_detections in detections.items():
        for obj_detection in obj_detections:
            marker = Marker()
            marker.header.frame_id = "velodyne"
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.id = marker_id
            marker.lifetime = rospy.Duration(3)  # Adjust the lifetime as needed

            # Set the marker pose and dimensions based on the detection
            marker.pose.position.x = obj_detection[0]
            marker.pose.position.y = obj_detection[1]
            marker.pose.position.z = obj_detection[2]
            marker.scale.x = obj_detection[3]
            marker.scale.y = obj_detection[4]
            marker.scale.z = obj_detection[5]

            # Calculate the quaternion from the yaw angle (obj_detection[6])
            quaternion = tf_transformations.quaternion_from_euler(0, 0, obj_detection[6])

            # Set the marker orientation using the computed quaternion
            marker.pose.orientation.x = quaternion[0]
            marker.pose.orientation.y = quaternion[1]
            marker.pose.orientation.z = quaternion[2]
            marker.pose.orientation.w = quaternion[3]

            # Set the marker color based on the object class
            color = colors[obj_class]
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)
            marker_id += 1
            
            # Get the position and orientation from obj_detection
            position_x, position_y, position_z = obj_detection[0], obj_detection[1], obj_detection[2]
            orientation_x, orientation_y, orientation_z, orientation_w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
            
            # Set the marker frame using the marker_id
            marker_frame = f"marker_frame_{marker_id}"
            reference_frame='velodyne'
            
            # Start the transformation publisher for this object in a new thread
            transform_thread = threading.Thread(target=publish_transform, args=(position_x, position_y, position_z, orientation_x, orientation_y, orientation_z, orientation_w, reference_frame, marker_frame))
            transform_thread.start()

    return marker_array


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/Model_fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    
    # Initialize the node
    rospy.init_node('lidar_pointcloud_listener', anonymous=True)
    marker_array_publisher = rospy.Publisher("detection_markers", MarkerArray, queue_size=1)
   

    configs = parse_test_configs()
    colors = {
    0: (255, 0, 0),   # class 0: red
    1: (0, 255, 0),   # class 1: green
    2: (0, 0, 255),   # class 2: blue
    "car": (255, 0, 0),  # Red
    "pedestrian": (0, 255, 0),  # Green
    "cyclist": (0, 0, 255)  # Blue
    # Add more colors for additional classes
}
    
    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    test_dataloader = create_test_dataloader(configs)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            metadatas, bev_maps, img_rgbs = batch_data
            input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
            t1 = time_synchronized()
            outputs = model(input_bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            # detections size (batch_size, K, 10)
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            t2 = time_synchronized()

            detections = detections[0]  # only first batch
            # Draw prediction in the image
            bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            bev_map = draw_predictions(bev_map, detections.copy(), configs.num_classes, show_confidence=True)

            # Rotate the bev_map
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

            #img_path = metadatas['img_path'][0]
            #img_rgb = img_rgbs[0].numpy()
            #img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
            #img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            #calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
            img_path = metadatas['img_path'][0]
            calib_path = img_path.replace(".png", ".txt").replace("image_2", "calib")
            calib = Calibration(calib_path)
            kitti_dets = convert_det_to_real_values(detections)
            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                #img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)

            #out_img = merge_rgb_to_bev(img_bgr, bev_map, output_width=configs.output_width)
            out_img =bev_map


            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,1 / (t2 - t1)))
            if configs.save_test_output:
                if configs.output_format == 'image':
                    img_fn = os.path.basename(metadatas['img_path'][0])[:-4]
                    cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), out_img)
                elif configs.output_format == 'video':
                    if out_cap is None:
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_cap = cv2.VideoWriter(
                            os.path.join(configs.results_dir, '{}.avi'.format(configs.output_video_fn)),
                            fourcc, 30, (out_cap_w, out_cap_h))
		
                    out_cap.write(out_img)
                else:
                    raise TypeError

            marker_array = create_marker_array(detections)
            marker_array_publisher.publish(marker_array)
            cv2.imshow('test-img', out_img)
            output_image_name = 'output_image_{}.png'.format(batch_idx)
            cv2.imwrite(output_image_name, out_img)
            print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
            if cv2.waitKey(0) & 0xFF == 27:
                break
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
