import torch
import numpy as np

class SphereSaliency():
    def __init__(self, num_drop, num_steps):
        self.a = num_drop # how many points to remove
        self.k = num_steps
        
        self.is_training = False
        self.count = np.zeros((NUM_CLASSES, ), dtype=bool)
        self.all_counters = np.zeros((NUM_CLASSES, 3), dtype=int)
        
        # The number of points is not specified
        self.pointclouds_pl, self.labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, None)
        self.is_training_pl = tf.placeholder(tf.bool, shape=())
        
        # simple model
        self.pred, self.end_points = MODEL.get_model(self.pointclouds_pl, self.is_training_pl)
        self.classify_loss = MODEL.get_loss(self.pred, self.labels_pl, self.end_points)
        #print(self.classify_loss)
        self.grad = tf.gradients(self.classify_loss, self.pointclouds_pl)[0]
        
        ## 3 folders to store all the situations
        if not os.path.exists(DUMP_DIR+'/pred_correct_adv_wrong'): os.mkdir(DUMP_DIR+'/pred_correct_adv_wrong')
        if not os.path.exists(DUMP_DIR+'/pred_wrong_adv_correct'): os.mkdir(DUMP_DIR+'/pred_wrong_adv_correct')
        if not os.path.exists(DUMP_DIR+'/pred_wrong_adv_wrong'): os.mkdir(DUMP_DIR+'/pred_wrong_adv_wrong')

    def compute_saliency(self, data):
        sphere_core = np.median(data, axis=1, keepdims=True)
            
        sphere_r = np.sqrt(np.sum(np.square(data - sphere_core), axis=2)) ## BxN
            
        sphere_axis = data - sphere_core ## BxNx3

        sphere_map = np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2), np.power(sphere_r, FLAGS.power))
        return sphere_map

        
    def drop_points(self, pointclouds_pl, labels_pl, sess):
        pointclouds_pl_adv = pointclouds_pl.copy()
        for i in range(self.k):
            grad = sess.run(self.grad, feed_dict={self.pointclouds_pl: pointclouds_pl_adv,
                                                  self.labels_pl: labels_pl,
                                                  self.is_training_pl: self.is_training})
            # change the grad into spherical axis and compute r*dL/dr
            ## mean value            
            #sphere_core = np.sum(pointclouds_pl_adv, axis=1, keepdims=True)/float(pointclouds_pl_adv.shape[1])
            ## median value
            sphere_core = np.median(pointclouds_pl_adv, axis=1, keepdims=True)
            
            sphere_r = np.sqrt(np.sum(np.square(pointclouds_pl_adv - sphere_core), axis=2)) ## BxN
            
            sphere_axis = pointclouds_pl_adv - sphere_core ## BxNx3

            if FLAGS.drop_neg:
                sphere_map = np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2), np.power(sphere_r, FLAGS.power))
            else:
                sphere_map = -np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2), np.power(sphere_r, FLAGS.power))

            drop_indice = np.argpartition(sphere_map, kth=sphere_map.shape[1]-self.a, axis=1)[:, -self.a:]

            tmp = np.zeros((pointclouds_pl_adv.shape[0], pointclouds_pl_adv.shape[1]-self.a, 3), dtype=float)
            for j in range(pointclouds_pl.shape[0]):
                tmp[j] = np.delete(pointclouds_pl_adv[j], drop_indice[j], axis=0) # along N points to delete
                
            pointclouds_pl_adv = tmp.copy()
            
        return pointclouds_pl_adv
    
    def plot_advsarial_samples(self, pointclouds_pl_adv, labels_pl, pred_val):
    
        for i in range(labels_pl.shape[0]):
        
            if labels_pl[i]!=pred_val[i] and not self.count[labels_pl[i]]:
            
                img_filename = 'label_%s_pred_%s.jpg' % (SHAPE_NAMES[labels_pl[i]],
                                                           SHAPE_NAMES[pred_val[i]])
                img_filename = os.path.join(DUMP_DIR, img_filename)
                
                pc_util.pyplot_draw_point_cloud(pointclouds_pl_adv[i], img_filename)
                
                self.count[labels_pl[i]] = True
                
    def plot_natural_and_advsarial_samples_all_situation(self, pointclouds_pl, pointclouds_pl_adv, labels_pl, pred_val, pred_val_adv):
        
        
        for i in range(labels_pl.shape[0]):
            if labels_pl[i] == pred_val[i]:
                if labels_pl[i] != pred_val_adv[i]:
                    img_filename = 'label_%s_advpred_%s_%d' % (SHAPE_NAMES[labels_pl[i]],
                                                              SHAPE_NAMES[pred_val_adv[i]], 
                                                              self.all_counters[labels_pl[i]][0])
                    self.all_counters[labels_pl[i]][0] += 1
                    img_filename = os.path.join(DUMP_DIR+'/pred_correct_adv_wrong', img_filename)
                    pc_util.pyplot_draw_point_cloud_nat_and_adv(pointclouds_pl[i], pointclouds_pl_adv[i], img_filename)    
            else:
                if labels_pl[i] == pred_val_adv[i]:
                    img_filename = 'label_%s_pred_%s_%d' % (SHAPE_NAMES[labels_pl[i]],
                                                              SHAPE_NAMES[pred_val[i]], 
                                                              self.all_counters[labels_pl[i]][1])
                    self.all_counters[labels_pl[i]][1] += 1        
                    img_filename = os.path.join(DUMP_DIR+'/pred_wrong_adv_correct', img_filename)
                    
                else:
            
                    img_filename = 'label_%s_pred_%s_advpred_%s_%d' % (SHAPE_NAMES[labels_pl[i]],
                                                              SHAPE_NAMES[pred_val[i]],
                                                              SHAPE_NAMES[pred_val_adv[i]],
                                                              self.all_counters[labels_pl[i]][2])
                    self.all_counters[labels_pl[i]][2] += 1
                    img_filename = os.path.join(DUMP_DIR+'/pred_wrong_adv_wrong', img_filename)
                
                pc_util.pyplot_draw_point_cloud_nat_and_adv(pointclouds_pl[i], pointclouds_pl_adv[i], img_filename)