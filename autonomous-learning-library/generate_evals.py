envs = [
"boxing_v1",
"ice_hockey_v1",
"joust_v2",
"mario_bros_v2",
"warlords_v2",
"foozpong_v1",
"pong_v1",
"combat_plane_v1",
"combat_tank_v1",
"double_dunk_v2",
"entombed_competitive_v2",
"entombed_cooperative_v2",
"quadrapong_v2",
"basketball_pong_v1",
"volleyball_pong_v1",
"flag_capture_v1",
"maze_craze_v2",
"space_invaders_v1",
"space_war_v1",
"surround_v1",
"tennis_v2",
"wizard_of_wor_v2",
]
four_p_envs = {
"warlords_v2",
"quadrapong_v2",
"volleyball_pong_v1",
"foozpong_v1",
}
agent_2p_list = ["first_0", "second_0"]
agent_4p_list = agent_2p_list + ["third_0", "fourth_0"]
for env in envs:
    agent_list = agent_4p_list if env in four_p_envs else agent_2p_list
    for checkpoint in range(200000, 20000000+200000, 400000):
        for vs_random in ["", "--vs-random"]:
            if vs_random:
                for agent in agent_list:
                    frames = 100000
                    print(f"workon main_env && python test_independent.py {env} {checkpoint} /home/ben/job_results/ --frames={frames} --agent={agent} {vs_random}")
            else:
                agent = "first_0"
                frames = 100000
                print(f"workon main_env && python test_independent.py {env} {checkpoint} /home/ben/job_results/ --frames={frames} --agent={agent} {vs_random}")
    checkpoint = 200000
    frames = 1000000
    print(f"workon main_env && python test_independent.py {env} {checkpoint} ~/job_results/ --frames={frames} --agent={agent} --vs-random --agent-random")
