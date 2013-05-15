from main import Main

m1 = Main()
#    print 'created main'

m1.do_submission = True
m1.regularisation = 'l1'
m1.penalty = 6.0
m1.classifier = 'multilogistic'

m1.do_submission = False
m1.val_percentage = 0.741522
m1.load_cached_data()
m1.reg_grid = [1,3,6,10,100,1000]

m1.train_model()
m1.predict()

#m1.eval_testset()
print 'done'
