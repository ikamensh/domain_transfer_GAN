from read_val import rainy_sim, sunny_sim, test_val
from deraNET import generator, gan, new_discriminator
import cv2
import numpy as np
from util import combine_images, restore
import random

n_pics = sunny_sim.shape[0]

g = generator()
g.load_weights('generator')

# for i in range(2):
#     ae.fit(rainy, sunny, epochs=4, batch_size=16 * i + 16)
#
#     sunny_rec = ae.predict(rainy[:50])
#     # sunny_rec = rainy_l[:500] + hi_net
#
#
#     sunny_rec5 = restore(sunny_rec[5])
#     cv2.imwrite("results/autoenc{}.jpg".format(i), np.concatenate([sunny[5], sunny_rec5, rainy[5]]))
#
#     test_rec = ae.predict(test_pics[:20])
#     # test_rec = test_l + test_net
#     cv2.imwrite("results/validation{}.jpg".format(i), np.concatenate([restore(test_rec[17]), test_val[17]]))
#     ae.save_weights('generator', True)


d = new_discriminator()
d.load_weights('discriminator')
d_on_g = gan(g, d)


BATCH_SIZE = 16
g_loss = 1
d_loss = 1

n_batches = int(n_pics/BATCH_SIZE)

for epoch in range(500):
        print("Epoch is", epoch)
        print("Number of batches", n_batches)
        for index in range(n_batches-2):

            if g_loss < 4.5:
                sunny_batch = sunny_sim[index*BATCH_SIZE:(index+3)*BATCH_SIZE]
                # rainy_batch = rainy_sim[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                noise = np.random.uniform(-1, 1, size=(BATCH_SIZE*3, 100))

                generated_images = g.predict(noise, verbose=0)
                # print(sunny_batch.shape)
                # print(generated_images.shape)
                X = np.concatenate((sunny_batch, generated_images))
                y = [1] * BATCH_SIZE*3 + [0] * BATCH_SIZE*3
                d_loss = d.fit(X,y, verbose=0).history['loss'][0]

                if index%30 == 0:
                    image = combine_images(X)
                    image = restore(image)
                    cv2.imwrite("results/mimage" + str(epoch) + ".jpg", image)

                print("batch %d d_loss : %s" % (index, str(d_loss)))

            if d_loss < 0.45:
                random_index = random.randint(0, n_batches-1)
                # rainy_batch = rainy_sim[random_index * BATCH_SIZE:(random_index + 1) * BATCH_SIZE]
                noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
                d.trainable = False
                g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
                d.trainable = True
                print("batch %d g_loss : %f" % (index, g_loss))

            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)

        # random_index = random.randint(0, n_batches - 1)
        # rainy_batch = rainy_sim[random_index * BATCH_SIZE:(random_index + 1) * BATCH_SIZE]
        if epoch % 25 == 0:
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            generated_images = g.predict(noise, verbose=0)

            image = combine_images(generated_images)
            image = restore(image)
            cv2.imwrite("results/mimage" + str(epoch) + ".jpg", image)

            train_pic = restore(np.concatenate([sunny_sim[5], generated_images[5], rainy_sim[5]]))
            cv2.imwrite("results/g_autoenc{}.jpg".format(epoch), train_pic)
            # test_rec = ae.predict(test_pics[:20])
            # cv2.imwrite("results/g_validation{}.jpg".format(epoch),
            #             np.concatenate([restore(test_rec[17]), test_val[17]]))










