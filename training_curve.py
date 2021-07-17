from matplotlib import pyplot as plt


targets = [2, 3, 4, 5, 7, 8, 15, 16, 19, 26, 28]
targetlabels = ["Participant 2", "Participant 3", "Participant 4", "Participant 5", "Participant 7", "Participant 8",
                "Participant 15", "Participant 16", "Participant 19", "Participant 26", "Participant 28"]
lineStyleList = ["-", "--", "-.", ":"]


def main():

    plt.figure()
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Accuracy (%)')

    hundred_array=[]
    for i in range(1,101):
        hundred_array.append(i)

    curv = 0
    for t in targets:
        log_path = "./log_training_bvp/{}.log".format(t)
        with open(log_path, 'r')as f:
            lines = f.readlines()

        for line in lines :
            nums = line.split('=')[1].split(',')
            array = []
            for num in nums:
                num1 = num.replace('[','')
                num1 = num1.replace(']', '')
                array.append(float(num1)*100)

        plt.plot( hundred_array,array, lineStyle=lineStyleList[curv % len(lineStyleList)],label='%s ' % ( t))
        curv = curv+1

    plt.legend(targetlabels)
    plt.title("BVP Training Curve")
    plt.show()


main()


