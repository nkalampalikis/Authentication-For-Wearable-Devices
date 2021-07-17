from params import Parameters
from copy import deepcopy
from genome import Genome
from random import shuffle, randint
import pathlib2
from data_extraction import *
import numpy as np
import os
import csv
from test import test
from train import Trainer
import ast
#import matplotlib.pyplot as plt
import csv
import argparse
from test_data_sets.first_sessions import prepareFirst

#def run(params, model, input_generator, pos_pts, neg_pts):
    #th = TestHarness()
    #train_tensor, train_labels, weights, test_tensor, test_labels = input_generator(params, pos_pts,neg_pts, use_SMOTE=params.smote)
    #results = th.run_test(
    #        model,
    #        train_tensor,train_labels,weights,
    #        test_tensor,test_labels,
    #        params.epochs
    #)
    #return results

#import test_data_sets.bin_negative
#import test_data_sets.d_bin_negative
#from test import test



# targets = [9,10,11,12,14,24]
targets_test = [2, 3, 4, 5, 7, 8, 13, 15, 16, 19, 26, 28]


param = Parameters()


target_list = [2, 3, 4, 5, 7, 8, 15, 16, 19, 26, 28]  # 13


def run_model(p, model, threshold, sequence_length, target):


    data_set = prepareFirst(target)

    print(data_set)

    others = deepcopy(targets_test)
    others.remove(target)

    print("Getting data for training")
    train_data_format = [
            # (session, segment)
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
        ]
    pos = collect_segment_data(p, target, train_data_format)
    neg = []
    for o in others:
        neg += collect_segment_data(p, o, train_data_format)

    train_tensor = np.ndarray((
        len(pos)+len(neg),
        2,
        3,
        p.window_sz*p.hz
    ))
    train_labels = []

    print("==> Formatting tensor...", end='', flush=True)
    for r in range(0,len(neg)):
        train_tensor[r] = neg[r]
        train_labels.append(0)
    for r in range(0,len(pos)):
        train_tensor[r+len(neg)] = pos[r]
        train_labels.append(1)
    train_labels = np.array(train_labels)
    print("done")

    # Generate data weights
    print("==> Generating weights...", end='', flush=True)
    weights = {
        0 : 1.0,
        1 : float(len(neg)/len(pos))
    }
    print("done")

    print("Training...")
    model.fit(
                train_tensor, train_labels,
                batch_size=32,
                epochs=100,
                verbose=0,
                class_weight=weights,
                shuffle=True
            )

    print("Testing")
    results = test(
        p,
        model,
        # data_set[t],
        data_set,
        threshold,
        sequence_length
    )
    return results

def test_genome(g, log_file,roc_curve):
    p = param
    metrics = []
    with open(log_file, 'w') as f:
        f.write(str(g.traits))
        f.write("\n")
        #f.write("ID\tTAR\tTRR\tFAR\tFRR\n")
        f.write("ID\tTAR\tTRR\n")
        avg_tar = 0
        avg_trr = 0
        avg_far = 0
        avg_frr = 0
        for i in targets_test:
            f.write("%d\t" % i)
            #other = deepcopy(targets)
            #other.remove(i)
            m = g.generate_model(p);
            print("Target: ", i)
            #print("Negative Set: ", other)
            #results = run(p, m, generate_input, cpp(p,i), cnp(p,other)
            #roc_list =[]
            results = run_model(p, m, .5, 1, i)
            tar = results[0]
            print()
            print()
            print ("~~~~~~~~~~~~~~~~~~")
            print ("TAR", "TRR", results[0],results[1])
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            avg_tar+=tar
            trr = results[1]
            avg_trr+=trr
            far = 1 - trr
            avg_far+=far
            frr = 1 - tar
            avg_frr+=frr
            f.write("%0.2f\t%0.2f\t\n" % (tar,trr))
            f.flush()
            for i in range(len(results[2])):
                if len(metrics) ==0:
                    metrics.append(results[2][i])
                elif len(metrics)==len(results[2]):
                    for k in range(len(metrics[i])):
                        metrics[i][k] =metrics[i][k]+ results[2][i][k]
                else:
                    temp1 = []
                    for k in range(len(results[2][i])):
                        temp1.append(results[2][i][k])
                    metrics.append(temp1)
        avg_far /= len(targets_test)
        avg_frr /= len(targets_test)
        avg_tar /= len(targets_test)
        avg_trr /= len(targets_test)
        score = (avg_far ** 2) + (avg_frr ** 2)
        f.write("Score: %d\n" % score)
    roc = []
    for i in range(len(metrics)):
        temp = []
        if metrics[i][0] ==0 and metrics[i][3]==0:
            tar = 0
        else:
            tar = (metrics[i][0])/(metrics[i][0]+metrics[i][3])
        if metrics[i][2] == 0 and metrics[i][1]==0:
            far = 0
        else:
            far = (metrics[i][2])/(metrics[i][2]+metrics[i][1])
        temp.append(tar)
        temp.append(far)
        roc.append(temp)
    with open(roc_curve, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(roc)

    return score


def read_gen(params):
    if not os.path.isdir("./genetic_{}/".format(params.window_sz)):
        return 0
    dirs = os.listdir("./genetic_{}/".format(params.window_sz))
    return len(dirs)


def save_current_gen(current_generation, generation_idex):
    gen_arr = []
    if not os.path.isdir("./saved_gens"):
        pathlib2.Path('./saved_gens').mkdir(parents=True, exist_ok=True)

    for i in range(len(current_generation)):
        gen = []
        gen.append(current_generation[i][0])
        gen.append(current_generation[i][1].mutation_rate)
        gen.append(current_generation[i][1].traits)
        gen_arr.append(gen)

    gen_file = "./saved_gens/saved_generation{}.csv".format(generation_idex)
    print("size:", len(gen_arr))
    with open(gen_file, "w", newline='') as gen_csv:
        csvWriter = csv.writer(gen_csv)
        csvWriter.writerows(gen_arr)


def load_current_gen(generation_idex):
    current_gen = []
    gen_file = "./saved_gens/saved_generation{}.csv".format(generation_idex - 1)
    with open(gen_file, "r") as gen_csv:
        current_generation_args = list(csv.reader(gen_csv))
    for args in current_generation_args:
        gen = []
        gen.append(float(args[0]))
        traits = ast.literal_eval(args[2])
        genome = Genome(args[1], traits, False)
        gen.append(genome)
        current_gen.append(gen)

    return current_gen

#for testing saving current gen

# current_gen = []
# for i in range(20):
#     a =[]
#     a.append(56789.32)
#     genome = Genome()
#     a.append(genome)
#     current_gen.append(a)
#
# save_current_gen(current_gen, 1000)
# print("test1")
# current_gen_load = load_current_gen(1001)
# print("test2")
# for val in current_gen_load:
#     print(val)


def main():

    population_size = 20
    fittest = 0.25
    ogres = 3
    global param
    current_generation = []


    ########## ARGUMENT PARSER ############################
    parser = argparse.ArgumentParser(description="Run genetic algorithm")
    parser.add_argument("-g", help="enter the number of generations to be run at a time", type=int, nargs=1)
    parser.add_argument("-w", help="enter the window size that the program will run", type=int, nargs=1)
    args = parser.parse_args()
    arguments = vars(args)

    if arguments["w"] is not None:
        WINDOW_SIZE = arguments["w"][0]
        param = Parameters(WINDOW_SIZE)

    if arguments["g"] is not None:
        GENERATIONS = arguments["g"][0]
    else:
        GENERATIONS = 2




    print("Running with WINDOW SIZE = ", param.window_sz)
    print("PROGRAM WILL RUN FOR ", GENERATIONS, " GENERATION(s)")

    ################### START SEQUENCE FOR THE FIRST GENERATION #####################
    if read_gen(param) == 0:
        gen_finish = read_gen(param) + GENERATIONS
        print("Generating seed generation")
        pathlib2.Path('./genetic_{}/gen0'.format(param.window_sz)).mkdir(parents=True, exist_ok=True)
        for i in range(0, population_size):
            print(i)
            g = Genome()


            log_file = "./genetic_{}/gen0/{}.log".format(param.window_sz, i)
            roc_curve = "./genetic_{}/gen0/{}.csv".format(param.window_sz, i)
            score = test_genome(g, log_file,roc_curve)

            current_generation.append([score, g])
        save_current_gen(current_generation, 0)
    else:
        gen_finish = read_gen(param) + GENERATIONS
    gen_num_new = read_gen(param)

    ##################################################################################


    ################### CONTINUE SEQUENCE FOR REMAINING GENERATIONS ##################
    for generation_idex in range(gen_num_new, gen_finish):

        current_generation = load_current_gen(generation_idex)
        pathlib2.Path('./genetic_{}/gen{}'.format(param.window_sz, generation_idex)).mkdir(parents=True, exist_ok=True)
        print("Breeding generation %d" % generation_idex)
        print("Selecting parents")

        # Figure out who the fittest parents are
        print("Current Generation", current_generation)
        current_generation = sorted(current_generation, key=lambda e: e[0], reverse=True)
        print("Current Generation", current_generation)
        parents = []
        for i in range(0, round(population_size*fittest)):

            log_file = "./genetic_{}/gen{}/parent_{}.log".format(param.window_sz,generation_idex,i)

            with open(log_file, 'w') as f:
                f.write(str(current_generation[0][0]))
                f.write("\n")
            parents.append(current_generation.pop(0))

        # Select our random losers ("ogres")
        shuffle(current_generation)
        for i in range(0, ogres):

            log_file = "./genetic_{}/gen{}/ogre_{}.log".format(param.window_sz, generation_idex,i)

            with open(log_file, 'w') as f:
                f.write(str(current_generation[0][0]))
                f.write("\n")
            parents.append(current_generation.pop(0))
        # Using parents, create a new generation
        current_generation = deepcopy(parents)
        child_idex = 0
        while len(current_generation) < population_size:
            parent1 = randint(0, len(parents) - 1)
            parent2 = randint(0, len(parents) - 1)
            if parent1 == parent2:
                continue
            else:
                # Add one child from this couple
                child = parents[parent1][1].breed(parents[parent2][1])

                log_file = "./genetic_{}/gen{}/{}.log".format(param.window_sz, generation_idex,child_idex)
                roc_curve = "./genetic_{}/gen{}/{}.csv".format(param.window_sz, generation_idex,child_idex)

                child_idex += 1
                score = test_genome(child, log_file, roc_curve)
                current_generation.append([score, child])
        save_current_gen(current_generation, generation_idex)

    ################################################################################

    best = sorted(current_generation, key=lambda e: e[0])[0]
    print("Best genome found:")
    print(best[1].traits)


if __name__ == "__main__":
    main()
#
# def print_roc(gen,index):
#     x = []
#     y = []
#     with open('./genetic/gen{}/{}.csv'.format(gen,index), 'r') as csvfile:
#         plots = csv.reader(csvfile, delimiter=',')
#         for row in plots:
#             if len(row) == 0:
#                 continue
#             print(row)
#             y.append(float(row[0]))
#             x.append(float(row[1]))
#     plt.plot(x, y, marker='o')
#     plt.title('ROC Curve')
#     plt.xlabel('FAR')
#     plt.ylabel('TAR')
#     plt.show()

#print_roc(0,0)