import time
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

logwriter = LogWriter('./log')

with logwriter.mode("decoding") as writer:
    da_scalar = writer.scalar("decoder_A")
    db_scalar = writer.scalar("decoder_B")

with logwriter.mode("generation") as writer:
    ga_scalar = writer.scalar("generation_A")
    gb_scalar = writer.scalar("generation_B")

    cyclea_scalar = writer.scalar("cycle_A")
    idta_scalar = writer.scalar("idt_A")

    cycleb_scalar = writer.scalar("cycle_B")
    idtb_scalar = writer.scalar("idt_B")
# with logwriter.mode("A") as writer:

# with logwriter.mode("B") as writer:



for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        #visualizer.reset()
        print 'visulizer reset'
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % 10 == 0:
            errors = model.get_current_errors()
            da_scalar.add_record(total_steps, errors['D_A'])
            ga_scalar.add_record(total_steps, errors['G_A'])
            cyclea_scalar.add_record(total_steps, errors['Cyc_A'])
            db_scalar.add_record(total_steps, errors['D_B'])
            gb_scalar.add_record(total_steps, errors['G_B'])
            cycleb_scalar.add_record(total_steps, errors['Cyc_B'])
            idta_scalar.add_record(total_steps, errors['idt_A'])
            idtb_scalar.add_record(total_steps, errors['idt_B'])


        if total_steps % 10 == 0:
            visuals = model.get_current_visuals()
            print visuals

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            #visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            print 'visualizer display current result'

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            #visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            print 'visualizer print current errors', epoch, epoch_iter, errors
            # if opt.display_id > 0:
            #     visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
