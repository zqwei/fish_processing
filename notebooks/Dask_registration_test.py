"""
Using dask

@author: modified by Salma Elmalaki, Dec 2018

"""
import chunk

import numpy as np
import pandas as pd
import os, sys
from glob import glob
from skimage import io
import dask.array as da
import dask.array.image as dai
import dask_ndfilters as daf
import dask.dataframe as dd
import dask.bag as db
import dask
from numba import vectorize

import time
from os.path import exists
import random


dat_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/ProcessedData/'
cameraNoiseMat = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/gainMat/gainMat20180208'
root_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/10182018/Fish3-2/'
save_folder = dat_folder + '10182018/Fish3-2/DataSample/using_100_samples/'
sample_folder = '/nrs/scicompsoft/elmalakis/Takashi_DRN_project/10182018/Fish3-2/test_sample/'  # Currently 100 samples


#################################### Batch parallelization (work for certain number of frames then cause overflow)#######################################

def registration_bag(is_largefile=True):
    from pathlib import Path
    #from fish_proc.pipeline.preprocess import motion_correction
    from skimage.io import imread, imsave
    import dask_image.imread
    import time
    import multiprocessing as mp

    start_time = time.time()

    if not os.path.isfile(save_folder + '/imgDMotion.tif') and os.path.isfile(save_folder + '/motion_fix.npy'):
        if not os.path.isfile(save_folder + '/proc_registr.tmp'):
            Path(save_folder + '/proc_registr.tmp').touch()

            imgD = imread(save_folder + '/imgDNoMotion.tif').astype('float32')
            #imgD = dask_image.imread.imread(save_folder + '/imgDNoMotion.tif', nframes=100).astype('float32')
            #imgD = imgD.compute()
            print("--- %s seconds for reading using  imread---" % (
                    time.time() - start_time))

            fix = np.load(save_folder + '/motion_fix.npy').astype('float32')

            len_D = len(imgD) // 2

            num_partitions = mp.cpu_count()
            print("registration batch into: "+str(num_partitions))
            batch1 = db.from_sequence(imgD[:len_D], npartitions=num_partitions)
            batch1 = batch1.map(motion_correction_element, fix)
            result1 = batch1.compute()
            print("--- %s seconds for batch 1 compute using bag with workers---" % (
                    time.time() - start_time))

            imgDMotion1, imgDMotionVar1 = zip(*result1)
            #np.save(save_folder + '/imgDMotion1.npy' , imgDMotion1)
            np.save(save_folder + '/imgDMotionVar1.npy', imgDMotionVar1)

            #imgDMotion1 = None
            imgDMotionVar1 = None


            batch2 = db.from_sequence(imgD[len_D:], npartitions=num_partitions)
            #batch2 = db.from_sequence(imgD[len_D:])
            batch2 = batch2.map(motion_correction_element, fix)

            result2 = batch2.compute()
            print("--- %s seconds for batch 2 compute using bag with workers---" % (
                    time.time() - start_time))
            imgDMotion2, imgDMotionVar2 = zip(*result2)

            np.save(save_folder + '/imgDMotionVar2.npy', imgDMotionVar2)

            #imgDMotion1 = np.load(save_folder + '/imgDMotion1.npy')
            imgDMotion = np.concatenate((imgDMotion1, imgDMotion2), axis=0)

            imgDMotion1 = None
            imgDMotion2 = None
            imgDMotionVar1 = None

            print("--- %s seconds for registration before save---" % (
                    time.time() - start_time))

            #imgDMotion = np.array(imgDMotion, dtype=np.float32)
            imsave(save_folder + '/imgDMotion.tif', imgDMotion, compress=1)

            #np.save(save_folder + '/imgDMotionVar', imgDMotionVar)

            imgDMotion = None

            Path(save_folder + '/finished_registr.tmp').touch()

    print("--- %s seconds for registration_elementwise after save---" % (
            time.time() - start_time))
    return None


#################################### Batch manually parallelization #######################################

def batch(seq, fix):
    sub_results = []
    for image_frame in seq:
        sub_results.append(motion_correction_dask_array(image_frame, fix))
    return sub_results

def registration_batch(is_largefile=True):
    from pathlib import Path
    #from fish_proc.pipeline.preprocess import motion_correction
    from skimage.io import imread, imsave
    import dask_image.imread
    import time
    import multiprocessing as mp

    start_time = time.time()

    if not os.path.isfile(save_folder + '/imgDMotion.tif') and os.path.isfile(save_folder + '/motion_fix.npy'):
        if not os.path.isfile(save_folder + '/proc_registr.tmp'):
            Path(save_folder + '/proc_registr.tmp').touch()
            imgDMotion = []
            imgDMotionVar = []

            imgD = imread(save_folder + '/imgDNoMotion.tif').astype('float32')
            #imgD = dask_image.imread.imread(save_folder + '/imgDNoMotion.tif', nframes=100).astype('float32')
            print("--- %s seconds for reading using  imread---" % (
                    time.time() - start_time))

            fix = np.load(save_folder + '/motion_fix.npy').astype('float32')

            #num_partitions = mp.cpu_count()
            #print("registration batch into: "+str(num_partitions))

            batches = []

            for img_frame in range(0, len(imgD), 80): # in steps of 100
                result_batch = dask.delayed(batch)(imgD[img_frame: img_frame+80], fix)
                batches.append(result_batch)

            result = dask.compute(*batches)

            print("--- %s seconds for registration_elementwise before flaten---" % (
                    time.time() - start_time))

            flat_list = [item for sublist in result for item in sublist]


            print("--- %s seconds for registration_elementwise before zip---" % (
                    time.time() - start_time))

            imgDMotion, imgDMotionVar = zip(*flat_list)

            print("--- %s seconds for registration_elementwise before save---" % (
                    time.time() - start_time))

            imgDMotion = np.array(imgDMotion, dtype=np.float32)
            imsave(save_folder + '/imgDMotion.tif', imgDMotion, compress=1)

            np.save(save_folder + '/imgDMotionVar', imgDMotionVar)

            imgDMotion = None
            imgDMotionVar = None

            Path(save_folder + '/finished_registr.tmp').touch()

    print("--- %s seconds for registration_elementwise after save---" % (
            time.time() - start_time))
    return None

#################################### Dask imread (error problem with multiple outputs from map_blocks)########################################
def rigid_stacks_dask_array(move, fix=None, trans=None):
    # start_time = time.time()
    if move.ndim < 3:
        move = move[np.newaxis, :]

    trans_move = move.copy()
    move_list = []

    for nframe, move_ in enumerate(move):
        trans_affine = trans.estimate_rigid2d(fix, move_)
        trans_mat = trans_affine.affine
        trans_move[nframe] = trans_affine.transform(move_)
        move_list.append([trans_mat[0, 1] / trans_mat[0, 0], trans_mat[0, 2], trans_mat[1, 2]])

    # print("--- %s seconds for rigidstacks---" % (time.time() - start_time))  # --salma
    return trans_move, move_list


def motion_correction_dask_array(image_frame, fix):
    from fish_proc.imageRegistration.imTrans import ImAffine

    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.ss_sigma_factor = 1.0

    imgDMotion_element, imgDMotionVar_element = rigid_stacks_dask_array(image_frame, fix=fix, trans=trans)

    return imgDMotion_element, imgDMotionVar_element



def registration_dask_array(is_largefile=True):
    from pathlib import Path
    # from fish_proc.pipeline.preprocess import motion_correction
    from skimage.io import imread, imsave
    import dask_image.imread
    import time
    import multiprocessing as mp

    start_time = time.time()

    if not os.path.isfile(save_folder + '/imgDMotion.tif') and os.path.isfile(save_folder + '/motion_fix.npy'):
        if not os.path.isfile(save_folder + '/proc_registr.tmp'):
            Path(save_folder + '/proc_registr.tmp').touch()
            imgDMotion = []
            imgDMotionVar = []

            imgD = dask_image.imread.imread(save_folder + '/imgDNoMotion.tif', nframes=100).astype('float32')
            print("--- %s seconds for reading using dask imread---" % (
                    time.time() - start_time))

            fix = np.load(save_folder + '/motion_fix.npy').astype('float32')

            result = imgD.map_blocks(motion_correction_dask_array, fix, dtype='float32')    # result not same size as input
            result.visualize("registration dask array.svg")
            result = result.compute()

            print("--- %s seconds for registration_elementwise before zip---" % (
                    time.time() - start_time))

            imgDMotion, imgDMotionVar = zip(*result)

            print("--- %s seconds for registration_elementwise before save---" % (
                    time.time() - start_time))

            imgDMotion = np.array(imgDMotion, dtype=np.float32)
            imsave(save_folder + '/imgDMotion.tif', imgDMotion, compress=1)

            np.save(save_folder + '/imgDMotionVar', imgDMotionVar)

            imgDMotion = None
            imgDMotionVar = None

            Path(save_folder + '/finished_registr.tmp').touch()

    print("--- %s seconds for registration_elementwise after save---" % (
            time.time() - start_time))
    return None


#################################### Element Wise #######################################
def rigid_stacks_element(move_frame, fix=None, trans=None):

    trans_affine = trans.estimate_rigid2d(fix, move_frame)
    trans_mat = trans_affine.affine
    trans_move = trans_affine.transform(move_frame)
    move_var = [trans_mat[0, 1]/trans_mat[0, 0], trans_mat[0, 2], trans_mat[1, 2]]

    return trans_move, move_var


def motion_correction_element(image_frame, fix):
    from fish_proc.imageRegistration.imTrans import ImAffine

    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.ss_sigma_factor = 1.0

    imgDMotion_element, imgDMotionVar_element = rigid_stacks_element(image_frame, fix=fix, trans=trans)

    return imgDMotion_element, imgDMotionVar_element


def registration_elementwise(is_largefile=True):
    from pathlib import Path
    #from fish_proc.pipeline.preprocess import motion_correction
    from skimage.io import imread, imsave
    import time

    start_time = time.time()

    if not os.path.isfile(save_folder + '/imgDMotion.tif') and os.path.isfile(save_folder + '/motion_fix_.npy'):
        if not os.path.isfile(save_folder + '/proc_registr.tmp'):
            Path(save_folder + '/proc_registr.tmp').touch()
            imgDMotion = []
            imgDMotionVar = []

            imgD = imread(save_folder + '/imgDNoMotion.tif').astype('float32')

            fix = np.load(save_folder + '/motion_fix_.npy').astype('float32')

            for image_frame in imgD:
                 imgDMotion_element, imgDMotionVar_element = dask.delayed(motion_correction_element, pure=False, nout=2)(image_frame, fix)
                 imgDMotion.append(imgDMotion_element)
                 imgDMotionVar.append(imgDMotionVar_element)

            imgDMotion = dask.compute(*imgDMotion)
            imgDMotionVar = dask.compute(*imgDMotionVar)

            print("--- %s seconds for registration_elementwise before save---" % (
                    time.time() - start_time))

            imgDMotion = np.array(imgDMotion, dtype=np.float32)
            imsave(save_folder + '/imgDMotion.tif', imgDMotion, compress=1)

         #   np.save(save_folder+'/imgDMotion', imgDMotion)
            np.save(save_folder + '/imgDMotionVar', imgDMotionVar)

    print("--- %s seconds for registration_elementwise after save---" % (
            time.time() - start_time))
    return None


#################################### multiprocesses (original implementation) #######################################
def rigidStacks(move, fix=None, trans=None):
    #start_time = time.time()
    if move.ndim < 3:
        move = move[np.newaxis, :]

    trans_move = move.copy()
    move_list = []

    for nframe, move_ in enumerate(move):
        trans_affine = trans.estimate_rigid2d(fix, move_)
        trans_mat = trans_affine.affine
        trans_move[nframe] = trans_affine.transform(move_)
        move_list.append([trans_mat[0, 1]/trans_mat[0, 0], trans_mat[0, 2], trans_mat[1, 2]])

    #print("--- %s seconds for rigidstacks---" % (time.time() - start_time))  # --salma

    return trans_move, move_list



def motion_correction(imgD_, fix_, fishName, ext=''):
    from fish_proc.imageRegistration.imTrans import ImAffine
    from fish_proc.utils.np_mp import parallel_to_chunks
    from fish_proc.utils.memory import get_process_memory, clear_variables
    import time

    trans = ImAffine()
    trans.level_iters = [1000, 1000, 100]
    trans.ss_sigma_factor = 1.0
    #print('memory usage before processing -- ')
    #get_process_memory();
    #start_time = time.time() # --salma
    imgDMotion, imgDMotionVar = parallel_to_chunks(rigidStacks, imgD_, fix=fix_, trans=trans)
    #imgDMotion, imgDMotionVar = regidStacks(imgD_, fix=fix_, trans=trans)
    #print("--- %s seconds for parallel_to_chunks rigidstacks---" % (time.time() - start_time))  # --salma
    #print("--- %s seconds for rigidstacksDask---" % (time.time() - start_time))  # --salma

    # imgStackMotion, imgStackMotionVar = parallel_to_chunks(regidStacks, imgStack, fix=fix, trans=trans)
    # np.save('tmpData/imgStackMotion', imgStackMotion)
    # np.save('tmpData/imgStackMotionVar', imgStackMotionVar)
    start_time = time.time()  # --salma
    np.save(fishName+'/imgDMotion%s'%(ext), imgDMotion)
    np.save(fishName+'/imgDMotionVar%s'%(ext), imgDMotionVar)
    print("--- %s seconds for saving imgDMotion and imgDMotionVar---" % (time.time() - start_time))  # --salma

    #print('memory usage after processing -- ')
    #get_process_memory();
    #print('release memory')
    imgDMotion = None
    imgDMotionVar = None
    clear_variables((imgDMotion, imgDMotionVar))
    return None


def registration(is_largefile=True):
    from pathlib import Path
    #from fish_proc.pipeline.preprocess import motion_correction
    from skimage.io import imread, imsave
    import time

    start_time = time.time()
    if not os.path.isfile(save_folder + '/imgDMotion.tif') and os.path.isfile(save_folder + '/motion_fix.npy'):
        if not os.path.isfile(save_folder + '/proc_registr.tmp'):
            Path(save_folder + '/proc_registr.tmp').touch()

            imgD_ = imread(save_folder + '/imgDNoMotion.tif').astype('float32')

            fix_ = np.load(save_folder + '/motion_fix.npy').astype('float32')

            if is_largefile:
                len_D_ = len(imgD_) // 2

                motion_correction(imgD_[:len_D_], fix_, save_folder, ext='0')
                # get_process_memory();
                motion_correction(imgD_[len_D_:], fix_, save_folder, ext='1')

                s_ = [np.load(save_folder + '/imgDMotion%d.npy' % (__)) for __ in range(2)]
                s_ = np.concatenate(s_, axis=0).astype('float32')

                imsave(save_folder + '/imgDMotion.tif', s_, compress=1)

                os.remove(save_folder + '/imgDMotion0.npy')
                os.remove(save_folder + '/imgDMotion1.npy')
            else:
                motion_correction(imgD_, fix_, save_folder)

                s_ = np.load(save_folder + '/imgDMotion.npy').astype('float32')
                imsave(save_folder + '/imgDMotion.tif', s_, compress=1)
                s_ = None
                os.remove(save_folder + '/imgDMotion.npy')

            Path(save_folder + '/finished_registr.tmp').touch()

            print("--- %s seconds for registration normal --" % (
                    time.time() - start_time))  # --salma
    return None


######### Tests and main #########

## python Dask_registration_test.py
if __name__ == '__main__':
    if len(sys.argv)>1:
        ext = ''
        if len(sys.argv)>2:
            ext = sys.argv[2]
        eval(sys.argv[1]+f"({ext})")
    else:
        #registration()
        registration_batch()  # trible the time when batch is 100
        #registration_dask_array()
        #registration_bag()