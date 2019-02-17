from load_data import *
from nn_module import CCANN
from utils import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-gd', '--gpu_device', type=str, help='device(s) on GPU, default=0', default='0')
parser.add_argument('-rp', '--raw_data_path', type=str, help='path for loading raw data', default=None)
parser.add_argument('-rt', '--data_ratio', type=str, help='ratio of train:valid:test', default='8:1:1')
parser.add_argument('-pd', '--processed_data_dir', type=str, help='the directory for saving processed data if -rp not None. load processed data from this directory if -rp None', default=None)
parser.add_argument('-ed', '--embedding_dir', type=str, help='the directory for loading pre-trained embedding', default=None)
parser.add_argument('-em', '--embedding_dim', type=int, help='if embedding_dir not provided, default=50', default=50)
parser.add_argument('-se', '--save_embedding_dir', type=str, help='save embeddings dir, default=None', default=None)

parser.add_argument('-bs', '--batch_size', type=int, help='batch size, default=128', default=128)
parser.add_argument('-it', '--max_iter', type=int, help='max iteration number, default=50', default=50)
parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate, default=0.001', default=0.001)
parser.add_argument('-fn', '--factor_num', type=int, help='factor number of FM, default=10', default=10)
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
if args.raw_data_path is not None and args.processed_data_dir is None:
    sys.exit(get_now_time() + 'provide processed_data_dir for saving data')
elif args.raw_data_path is None and args.processed_data_dir is None:
    sys.exit(get_now_time() + 'provide processed_data_dir for loading data')

if args.raw_data_path is not None:
    ratio = [float(r) for r in args.data_ratio.split(':')]
    split_raw_data(args.raw_data_path, args.processed_data_dir, ratio[0], ratio[1], ratio[2])


if args.embedding_dir is None:
    train_tuple_list, valid_tuple_list, test_tuple_list, max_r, min_r, index2user, index2item, index2comp, index2month, \
    index2city = load_ccann_data_id(args.processed_data_dir)
    fm = CCANN(train_tuple_list, len(index2user), len(index2item), len(index2comp), len(index2month), len(index2city), args.embedding_dim, args.factor_num, args.batch_size, args.learning_rate)
else:
    user_embedding, index2user, user2index, item_embedding, index2item, item2index, comp_embedding, index2comp,\
    comp2index, month_embedding, index2month, month2index, city_embedding, index2city, city2index = load_embeddings(args.embedding_dir)
    train_tuple_list, valid_tuple_list, test_tuple_list, max_r, min_r = load_ccann_data(args.processed_data_dir, user2index, item2index, comp2index, month2index, city2index)
    fm = CCANN(train_tuple_list, user_embedding, item_embedding, comp_embedding, month_embedding, city_embedding, None, args.factor_num, args.batch_size, args.learning_rate)


# early stop setting
pre_val_eval = 1e10
for it in range(1, args.max_iter + 1):
    print(get_now_time() + 'iteration ' + str(it))
    fm.train_one_epoch()

    # evaluating
    train_predict = fm.get_prediction(train_tuple_list)
    train_rmse = root_mean_square_error(train_predict, max_r, min_r)
    print(get_now_time() + 'RMSE on train data: ' + str(train_rmse))
    valid_predict = fm.get_prediction(valid_tuple_list)
    valid_rmse = root_mean_square_error(valid_predict, max_r, min_r)
    print(get_now_time() + 'RMSE on valid data: ' + str(valid_rmse))
    test_predict = fm.get_prediction(test_tuple_list)
    test_rmse = root_mean_square_error(test_predict, max_r, min_r)
    print(get_now_time() + 'RMSE on test data: ' + str(test_rmse))

    # early stop setting
    if valid_rmse > pre_val_eval:
        print(get_now_time() + 'early stopped')
        break
    pre_val_eval = valid_rmse


if args.save_embedding_dir is not None:
    user_embeddings, item_embeddings, comp_embeddings, month_embeddings, city_embeddings = fm.get_embeddings()
    user2embeddings = {}
    item2embeddings = {}
    comp2embeddings = {}
    month2embeddings = {}
    city2embeddings = {}
    for idx, vec in enumerate(user_embeddings):
        u = index2user[idx]
        user2embeddings[u] = vec
    for idx, vec in enumerate(item_embeddings):
        i = index2item[idx]
        item2embeddings[i] = vec
    for idx, vec in enumerate(comp_embeddings):
        p = index2comp[idx]
        comp2embeddings[p] = vec
    for idx, vec in enumerate(month_embeddings):
        m = index2month[idx]
        month2embeddings[m] = vec
    for idx, vec in enumerate(city_embeddings):
        c = index2city[idx]
        city2embeddings[c] = vec
    pickle.dump(user2embeddings, open(args.save_embedding_dir + 'user2embeddings.pickle', 'wb'))
    pickle.dump(item2embeddings, open(args.save_embedding_dir + 'item2embeddings.pickle', 'wb'))
    pickle.dump(comp2embeddings, open(args.save_embedding_dir + 'comp2embeddings.pickle', 'wb'))
    pickle.dump(month2embeddings, open(args.save_embedding_dir + 'month2embeddings.pickle', 'wb'))
    pickle.dump(city2embeddings, open(args.save_embedding_dir + 'city2embeddings.pickle', 'wb'))

    print(get_now_time() + 'saved embeddings')
