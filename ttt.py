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
            
            #sphere_map = -np.sum(np.multiply(grad, sphere_axis), axis=2) ## BxN
            sphere_map = -np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2), np.power(sphere_r, 6))

            drop_indice = np.argpartition(sphere_map, kth=sphere_map.shape[1]-self.a, axis=1)[:, -self.a:]
            #print(drop_indice[0:2], np.argmax(sphere_map, axis=1)[0:2])

            tmp = np.zeros((pointclouds_pl_adv.shape[0], pointclouds_pl_adv.shape[1]-self.a, 3), dtype=float)
            for j in range(pointclouds_pl.shape[0]):
                tmp[j] = np.delete(pointclouds_pl_adv[j], drop_indice[j], axis=0) # along N points to delete
                
            pointclouds_pl_adv = tmp.copy()
            
        return pointclouds_pl_adv