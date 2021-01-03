class MultiAgentEnv(object):
    """
    MultiAgent env abstract class
    """
    def reset(self):
        """
        Resets envs and returns observations from agents present in the env.
        Returns:
            obs (dict): New observations for each present agent.
        """
        raise NotImplementedError

    def step(self, action_dict):
        """
        Steps the environment forward one step based on the actions provided in action_dict.
        Args:
            action_dict (dict): action for each ready agent
        Returns:
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        raise NotImplementedError