
from director.embodied.envs import loconav as nav 
MAPS = {

    'maze_s': (
        '            6 6 6 6 6',
        '            6 . . . 6',
        '            6 . P . 6',
        '            6 . . . 6',
        '            5 . . . 4',
        '            5 . . . 4',
        '1 1 1 1 5 5 5 . . . 4',
        '1 . . . . . . . . . 3',
        '1 . P . . . . . . . 3',
        '1 . . . . . . . . . 3',
        '1 1 1 1 2 2 2 3 3 3 3',
    ),

    'maze_m': (
        '6 6 6 6 8 8 8 7 7 7 7',
        '6 . . . . . . . . . 7',
        '6 . P . . . P . . . 7',
        '6 . . . . . . . . . 7',
        '6 6 6 5 5 5 5 . . . 4',
        '            5 . P . 4',
        '1 1 1 1 5 5 5 . . . 4',
        '1 . . . . . . . . . 3',
        '1 . P . . . P . . . 3',
        '1 . . . . . . . . . 3',
        '1 1 1 1 2 2 2 3 3 3 3',
    ),

    # 'maze_m': (
    #     '8 8 8 8 9 9 9 A A A B',
    #     '7 . . . . . . . . . B',
    #     '7 . . P . . . . . . B',
    #     '7 . . . . . . . . . B',
    #     '6 6 6 5 5 5 5 . . . C',
    #     '            4 . P . C',
    #     '2 2 2 2 3 3 3 . . . C',
    #     '1 . . . . . . . . . D',
    #     '1 . P . . . . . . . D',
    #     '1 . . . . . . . . . D',
    #     'F F F F F E E E E E D',
    # ),

    'maze_l': (
        '8 8 8 8 7 7 7 6 6 6 6 . . .',
        '8 . . . . . . . . . 6 . . .',
        '8 . . . . P . . . . 6 . . .',
        '8 . . . . . . . . . 6 5 5 5',
        '8 8 8 8 7 7 7 . . . . . . 5',
        '. . . . . . 7 . . . . . . 5',
        '1 1 1 1 1 . 7 . . . . . . 5',
        '1 . . . 1 . 7 9 9 9 . P . 5',
        '1 . . . 1 . . . . 9 . . . 5',
        '1 . . . 1 1 1 9 9 9 . . . 5',
        '2 . . . . . . . . . . . . 4',
        '2 . . . . P . . . . . . . 4',
        '2 . . . . . . . . . . . . 4',
        '2 2 2 2 3 3 3 3 3 3 4 4 4 4',
    ),

    'maze_xl': (
        '9 9 9 9 9 9 9 8 8 8 8 . 4 4 4 4 4',
        '9 . . . . . . . . . 8 . 4 . . . 4',
        '9 . . . P . . . . . 8 . 4 . . . 4',
        '9 . . . . . . . . . 8 . 4 . P . 4',
        '6 . . . 7 7 7 8 8 8 8 . 5 . . . 3',
        '6 . P . 7 . . . . . . . 5 . . . 3',
        '6 . . . 7 7 7 5 5 5 5 5 5 . P . 3',
        '5 . . . . . . . . . . . . . . . 3',
        '5 . P . . . . . P . . . . . P . 3',
        '5 . . . . . . . . . . . . . . . 3',
        '5 5 5 5 4 4 4 . . . 6 6 6 . . . 3',
        '. . . . . . 4 . . . 6 . 6 . P . 3',
        '1 1 1 1 4 4 4 . P . 6 . 6 . . . 3',
        '1 . . . . . . . . . 2 . 1 . . . 1',
        '1 . P . . P . . . . 2 . 1 . P . 1',
        '1 . . . . . . . . . 2 . 1 . . . 1',
        '1 1 1 1 1 1 1 2 2 2 2 . 1 1 1 1 1',
    ),

    'maze_xxl': (
        '7 7 7 7 * * * 6 6 6 * * * 9 9 9 9',
        '7 . . . . . . . . . . . . . . . 9',
        '7 . . . P . . . . . . P . . . . 9',
        '7 . . . . . . . . . . . . . . . 9',
        '* . . . 5 5 5 * * * * * * 9 9 9 9',
        '* . . . 5 . . . . . . . . . . . .',
        '* . P . 5 5 5 * * * * * * 3 3 3 3',
        '8 . . . . . . . . . . . . . . . 3',
        '8 . . . . . P . . . . P . . . . 3',
        '8 . . . . . . . . . . . . . . . 3',
        '8 8 8 8 * * * * * * 4 4 4 . . . *',
        '. . . . . . . . . . . . 4 . P . *',
        '1 1 1 1 * * * * * * 4 4 4 . . . *',
        '1 . . . . . . . . . . . . . . . 2',
        '1 . P . . . P . . . . P . . . . 2',
        '1 . . . . . . . . . . . . . . . 2',
        '1 1 1 1 * * * 6 6 6 * * * 2 2 2 2',
    ),

    'empty': (
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
    ),

}


class EgocentricMaze(nav.LocoNav):

    def __init__(
      self, name, goal, repeat=1, size=(64, 64), camera=-1, again=False,
      termination=False, weaker=1.0):
        self._goal = goal
        super().__init__(name, repeat, size, camera, again, termination, weaker)

    def _make_arena(self, name):
        import labmaze
        from dm_control import mjcf
        from dm_control.locomotion.arenas import labmaze_textures
        from dm_control.locomotion.arenas import mazes
        import matplotlib.pyplot as plt
        class WallTexture(labmaze_textures.WallTextures):
            def _build(self, color=[0.8, 0.8, 0.8], model='labmaze_style_01'):
                self._mjcf_root = mjcf.RootElement(model=model)
                self._textures = [self._mjcf_root.asset.add(
                    'texture', type='2d', name='wall', builtin='flat',
                    rgb1=color, width=100, height=100)]
        wall_textures = {'*': WallTexture([0.8, 0.8, 0.8])}
        cmap = plt.get_cmap('tab20')
        for i, index in enumerate(list(range(9)) + ['A', 'B', 'C', 'D', 'E', 'F']):
            if isinstance(index, int):
                wall_textures[str(i + 1)] = WallTexture(cmap(i)[:3])
            else:
                wall_textures[index] = WallTexture(cmap(i)[:3])


        lines = [line[::2].replace('.', ' ') for line in MAPS[name]]
        lines[self._goal[0]] = lines[self._goal[0]][:self._goal[1]] + 'G' + lines[self._goal[0]][self._goal[1]+1:]

        layout = ''.join([
            l + '\n' for l in lines])
        
        print(layout)
        maze = labmaze.FixedMazeWithRandomGoals(
            entity_layer=layout,
            num_spawns=None, num_objects=1, random_state=None)
        arena = mazes.MazeWithTargets(
            maze, xy_scale=1.2, z_height=2.0, aesthetic='default',
            wall_textures=wall_textures, name='maze')
        
        
        return arena


   
