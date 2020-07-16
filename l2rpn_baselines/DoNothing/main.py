import grid2op
from l2rpn_baselines.DoNothing import evaluate


env = grid2op.make("D:\\Projets\\RTE\\ExpertOp4Grid\\alphaDeesp\\ressources\\parameters\\l2rpn_2019_ltc_9")
res = evaluate(env)
