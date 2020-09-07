import grid2op
from l2rpn_baselines.ExpertAgent import evaluate


env = grid2op.make("D:\\Projets\\RTE\\ExpertOp4Grid\\alphaDeesp\\ressources\\parameters\\l2rpn_neurips_2020_track1_val")
res = evaluate(env)
