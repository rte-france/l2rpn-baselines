import grid2op
from l2rpn_baselines.ExpertAgent import evaluate


env = grid2op.make("D:\\RTE\\ExpertOp4Grid\\1 - Développement\\ExpertOp4Grid\\alphaDeesp\\ressources\\parameters\\l2rpn_2019_ltc_9")
res = evaluate(env)
