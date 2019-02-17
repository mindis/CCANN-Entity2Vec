from module import CBOW
from load_data import *
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-gd', '--gpu_device', type=str, help='device(s) on GPU, default=\'0\'', default='0')
parser.add_argument('-rp', '--raw_data_path', type=str, help='path for loading raw data, should be provided', default=None)
parser.add_argument('-ed', '--embedding_dir', type=str, help='the directory for saving pre-trained embedding', default=None)

parser.add_argument('-bs', '--batch_size', type=int, help='batch size, default=256', default=256)
parser.add_argument('-it', '--max_iter', type=int, help='max iteration number, default=50', default=50)
parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate, default=0.001', default=0.001)
parser.add_argument('-es', '--embedding_size', type=int, help='embedding size, default=100', default=100)
parser.add_argument('-ws', '--window_size', type=int, help='window size, default=1', default=1)
parser.add_argument('-ns', '--num_sampled', type=int, help='negative sample size, default=64', default=64)
parser.add_argument('-mw', '--max_word_num', type=int, help='max word num, default=20000', default=20000)
parser.add_argument('-sr', '--sample_ratio', type=float, help='ratio for sampling word-target pairs from a review, default=0.1', default=0.1)
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
if args.raw_data_path is None:
    sys.exit(get_now_time() + 'raw_data_path cannot be None')
if args.embedding_dir is None:
    sys.exit(get_now_time() + 'embedding_dir cannot be None')


u_i_r_p_m_c_tuple_list, index2word = load_raw_data(args.raw_data_path, args.max_word_num)
u_i_r_p_m_c_tuple_list, index2user, index2item, index2comp, index2month, index2city = format_train_data(u_i_r_p_m_c_tuple_list)
cbow = CBOW(u_i_r_p_m_c_tuple_list, len(index2word), len(index2user), len(index2item), len(index2comp), len(index2month),
            len(index2city), args.batch_size, args.embedding_size, args.window_size, args.num_sampled, args.learning_rate,
            args.sample_ratio)


# early stop setting
pre_val_eval = 1e10
for it in range(1, args.max_iter + 1):
    print(get_now_time() + 'iteration {}'.format(it))

    loss = cbow.train_one_epoch()
    print(get_now_time() + 'loss in train data: {}'.format(loss))

    # early stop setting
    if loss > pre_val_eval:
        print(get_now_time() + 'early stopped')
        break
    pre_val_eval = loss


# save embedding
word_embeddings, user_embeddings, item_embeddings, comp_embeddings, month_embeddings, city_embeddings = cbow.get_embeddings()
user2embeddings = {}
item2embeddings = {}
comp2embeddings = {}
month2embeddings = {}
city2embeddings = {}
word2embeddings = {}

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
for idx, vec in enumerate(word_embeddings):
    w = index2word[idx]
    word2embeddings[w] = vec

if not os.path.exists(args.embedding_dir):
    os.makedirs(args.embedding_dir)
pickle.dump(user2embeddings, open(args.embedding_dir + 'user2embeddings.pickle', 'wb'))
pickle.dump(item2embeddings, open(args.embedding_dir + 'item2embeddings.pickle', 'wb'))
pickle.dump(comp2embeddings, open(args.embedding_dir + 'comp2embeddings.pickle', 'wb'))
pickle.dump(month2embeddings, open(args.embedding_dir + 'month2embeddings.pickle', 'wb'))
pickle.dump(city2embeddings, open(args.embedding_dir + 'city2embeddings.pickle', 'wb'))
pickle.dump(word2embeddings, open(args.embedding_dir + 'word2embeddings.pickle', 'wb'))

print(get_now_time() + 'saved embeddings')
