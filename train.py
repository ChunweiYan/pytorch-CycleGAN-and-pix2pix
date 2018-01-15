import time
import random
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from visualdl import LogWriter
#from util.visualizer import Visualizer

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
#visualizer = Visualizer(opt)
total_steps = 0

logwriter = LogWriter('./log', 20)

with logwriter.mode("real") as writer:
    reala_image = writer.image("left/real_A", 10, 1)
    fakea_image = writer.image("left/fake_A", 10, 1)


with logwriter.mode("fake") as writer:
    da_scalar = writer.scalar("right/decoder_A")
    db_scalar = writer.scalar("right/decoder_B")

with logwriter.mode("fake") as writer:
    ga_scalar = writer.scalar("left/generation_A")
    gb_scalar = writer.scalar("right/generation_B")

with logwriter.mode("real") as writer:
    cyclea_scalar = writer.scalar("left/cycle_A")
    idta_scalar = writer.scalar("left/idt_A")

with logwriter.mode("fake") as writer:
    cycleb_scalar = writer.scalar("right/cycle_B")
    idtb_scalar = writer.scalar("right/idt_B")


scalar_counter = 0.
image_step_cycle = 20
image_start = False

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    print 'epoch', epoch_iter

    for i, data in enumerate(dataset, 50):
        iter_start_time = time.time()

        if total_steps % image_step_cycle == 0:
            print 'image start sampling'
            reala_image.start_sampling()
            fakea_image.start_sampling()
            image_start = True


        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % 2 == 0:
            print 'model', model.netG_B.model.state_dict()
            errors = model.get_current_errors()
            da_scalar.add_record(total_steps, errors['D_A'])
            ga_scalar.add_record(total_steps, errors['G_A'])
            cyclea_scalar.add_record(total_steps, errors['Cyc_A'])
            db_scalar.add_record(total_steps, errors['D_B'])
            gb_scalar.add_record(total_steps, errors['G_B'])
            cycleb_scalar.add_record(total_steps, errors['Cyc_B'])
            idta_scalar.add_record(total_steps, errors['idt_A'])
            idtb_scalar.add_record(total_steps, errors['idt_B'])


        if image_start and total_steps % 2 == 0:
            visuals = model.get_current_visuals()

            idx = reala_image.is_sample_taken()
            print 'idx:', idx
            if idx != -1:
                data = visuals['real_A']
                reala_image.set_sample(idx, data.shape, data.flatten())

            idx = fakea_image.is_sample_taken()
            if idx != -1:
                data = visuals['fake_A']
                fakea_image.set_sample(idx, data.shape, data.flatten())


        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

        if total_steps % image_step_cycle == 0:
            reala_image.finish_sampling()
            fakea_image.finish_sampling()
            image_start = False

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

