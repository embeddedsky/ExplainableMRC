from RC_CRJ import  *
import sys


if __name__=="__main__":
    # argv:
    # 1.RC方法名：RC_CRJ ,RC_RND,RC_CND,RC_DCND
    # 2.是否自适应: "1"--自适应，"0"--不自适应
    # 3.优化问题: "Narma","Three_Out"
    # 4.模型：CHEN ,HP

    RC_Method,Is_Adapt,Problem,Model=sys.argv[1:]
    # three_signal_one.create_signal(Problem+"/"+Train_or_Test)
    rc=eval(RC_Method)(Is_Adapt)
    rc.evolve(200)

