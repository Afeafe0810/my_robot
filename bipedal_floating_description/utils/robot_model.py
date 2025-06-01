import numpy as np; np.set_printoptions(precision=2)
import pinocchio as pin #2.6.21
import pink #2.1.0

#================ import other code =====================#
from utils.config import Config
#========================================================#
    
class RobotModel:
    """
    一個使用 Pinocchio 和 Meshcat 進行可視化的機器人模型處理類別。
    - 屬性:
        - bipedal_floating: 從骨盆建下來的模擬模型。
        - single_lf: 從左腳掌往上建的左單腳。
        - single_rf: 從右腳掌往上建的右單腳。
        - bipedal_from_lf: 從左腳掌建起的雙腳。
        - bipedal_from_rf: 從右腳掌建起的雙腳。
    - 方法:
        - update_VizAndMesh: 給定關節轉角，更新機器人模型，回傳機器人的 configuration。
    """
    def __init__(self):
        #=========建立機器人模型===========#
        self._meshrobot = self._loadMeshcatModel("/bipedal_floating.pin.urdf") #Pinnocchio藍色的機器人

        self.bipedal_from_pel = BipedalFromPel("/bipedal_floating.xacro") #從骨盆建下來的模擬模型
        self.bipedal_from_lf = BipedalFromLF("/bipedal_l_gravity.xacro") #從左腳掌建起的雙腳
        self.bipedal_from_rf = BipedalFromRF("/bipedal_r_gravity.xacro") #從右腳掌建起的雙腳
        self.single_lf = SingleLeftLeg("/stance_l.xacro") #從左腳掌往上建的左單腳
        self.single_rf = SingleRightLeg("/stance_r_gravity.xacro") #從右腳掌往上建的右單腳
        
        #=========可視化msehcat===========#
        self._viz = self._meshcatVisualize(self._meshrobot)
        self.update_VizAndMesh(self._meshrobot.q0) #可視化模型的初始關節角度
        
    def update_VizAndMesh(self, jp: np.ndarray) -> pink.Configuration:
        '''給定關節轉角, 更新機器人模型, 回傳機器人的configuration'''
        config = pink.Configuration(self._meshrobot.model, self._meshrobot.data, jp)
        self._viz.display(config.q)
        return config
    
    def gravity(self, jp: np.ndarray, state: float, stance: list[str], pa_lfTOpel_in_pf: np.ndarray, pa_rfTOpel_in_pf: np.ndarray) -> np.ndarray:
        
        def weighted(x, x0, x1, g0, g1) -> np.ndarray:
            return g0 +(g1-g0)/(x1-x0)*(x-x0)
            
        #==========半邊單腳模型==========#
        g_from_both_single_ft = np.vstack((
            self.single_lf.gravity(jp),
            self.single_rf.gravity(jp)
        ))
        
        #==========腳底建起的模型==========#
        g_from_bipedal_ft = {
            'lf': self.bipedal_from_lf.gravity(jp),
            'rf': self.bipedal_from_rf.gravity(jp)
        }
        
        match state:
            case 0 | 1 | 2 | 3:
                y_ftTOpel = {'lf': abs(pa_lfTOpel_in_pf[1,0]), 'rf': abs(pa_rfTOpel_in_pf[1,0])}
                
                #雙腳平衡時, 用距離判斷重心腳
                cf = 'lf' if y_ftTOpel['lf'] <= y_ftTOpel['rf'] else 'rf'
                
                return weighted(y_ftTOpel[cf], *[0, 0.1], *[g_from_bipedal_ft[cf], g_from_both_single_ft])
                
            case 30:
                cf, sf = stance
                
                return 0.3 * g_from_both_single_ft + 0.75* g_from_bipedal_ft[cf]
    
    def inertia(self, jp: np.ndarray, stance: list[str]) -> np.ndarray:
        cf, sf = stance
        inertia_from_ft = {'lf': self.bipedal_from_lf.inertia, 'rf': self.bipedal_from_rf.inertia}
        return inertia_from_ft[cf](jp)
    
    def pure_knee_inertia(self, jp: np.ndarray, stance: list[str]) -> np.ndarray:
        total_inertia = self.inertia(jp, stance)
        
        the_knee = [0, 1, 2, 3, 6, 7, 8, 9]
        the_ankle = [4, 5, 10, 11]
        
        H_kneeTOknee = total_inertia[np.ix_(the_knee, the_knee)]
        H_ankleTOknee = total_inertia[np.ix_(the_knee, the_ankle)]
        H_kneeTOankle = total_inertia[np.ix_(the_ankle, the_knee)]
        H_ankleTOankle = total_inertia[np.ix_(the_ankle, the_ankle)]
        
        return H_kneeTOknee - H_ankleTOknee @ np.linalg.pinv(H_ankleTOankle) @ H_kneeTOankle
        
    @staticmethod
    def _loadMeshcatModel(urdf_path: str):
        '''高級動力學模型'''
        robot = pin.RobotWrapper.BuildFromURDF(
            filename = Config.ROBOT_MODEL_DIR + urdf_path,
            package_dirs = ["."],
            root_joint=None,
        )
        print(f"URDF description successfully loaded in {robot}")
        print(robot.model)
        print(robot.q0)
        return robot
    
    @staticmethod
    def  _meshcatVisualize(meshrobot: pin.RobotWrapper):
        '''可視化高級動力學模型'''
        viz = pin.visualize.MeshcatVisualizer( meshrobot.model, meshrobot.collision_model, meshrobot.visual_model )
        meshrobot.setVisualizer(viz, init=False)
        viz.initViewer(open=True)
        viz.loadViewerModel()

        return viz

class _AbstractSimpleModel:
    '''基礎運動學模型, 用來算重力矩和質心位置'''
    def __init__(self, urdf_path: str):
        #機器人模型
        self.model = pin.buildModelFromUrdf(Config.ROBOT_MODEL_DIR + urdf_path)
        self.data = self.model.createData()
        print(f'model: {self.model.name}')
        
        #關節順序的轉換
        self.permut: np.ndarray = self._joint_permutation()
        self.inv_permut: np.ndarray = self._joint_inverse_permutation()
        
    @staticmethod
    def _joint_permutation()-> np.ndarray:
        """要實作把關節順序/方向轉成urdf的順序"""
        raise NotImplementedError
    
    def _joint_inverse_permutation(self)-> np.ndarray:
        """把urdf的順序轉成關節順序/方向
        
        預設是12->12, 置換「方陣」的反函數為轉置, 非12->12需要重做"""
        
        return self._joint_permutation().T
    
    def gravity(self, jp: np.ndarray) -> np.ndarray:
        return self.inv_permut @ np.vstack(pin.computeGeneralizedGravity(self.model, self.data, self.permut@jp))
    
    def inertia(self, jp: np.ndarray) -> np.ndarray:
        return self.inv_permut @ pin.crba(self.model, self.data, self.permut@jp) @ self.permut
    
class BipedalFromPel(_AbstractSimpleModel):
    @staticmethod
    def _joint_permutation()-> np.ndarray:
        return np.eye(12)
    
    def com(self, jp: np.ndarray) -> np.ndarray:
        # TODO 學長有用其他模型建立, 不知道會不會有差, 但我目前是覺得就算有差也不可能差多少啦
        pin.centerOfMass(self.model, self.data, jp)
        return self.data.com[0].reshape(3,1)

class BipedalFromLF(_AbstractSimpleModel):
    @staticmethod
    def _joint_permutation()-> np.ndarray:
        """關節順序轉成5-0、6-11, 反轉須反向"""
        P = _pure_permute()
        O = np.zeros((6,6))
        I = np.eye(6,6)
        return np.block([[P, O], [O, I]])
    
class BipedalFromRF(_AbstractSimpleModel):
    @staticmethod
    def _joint_permutation()-> np.ndarray:
        """關節順序轉成11-6、0-5, 反轉須反向"""
        P = _pure_permute()
        O = np.zeros((6,6))
        I = np.eye(6,6)
        return np.block([[O, P], [I, O]])

class SingleLeftLeg(_AbstractSimpleModel):
    @staticmethod
    def _joint_permutation()-> np.ndarray:
        """關節順序轉成5-0, 反轉須反向"""
        P = _pure_permute()
        O = np.zeros((6,6))
        return np.block([P, O])
    
    @staticmethod
    def _joint_inverse_permutation():
        return _pure_permute()
    
class SingleRightLeg(_AbstractSimpleModel):
    @staticmethod
    def _joint_permutation()-> np.ndarray:
        """關節順序轉成11-6, 反轉須反向"""
        P = _pure_permute()
        O = np.zeros((6,6))
        return np.block([O, P])
    
    @staticmethod
    def _joint_inverse_permutation():
        return _pure_permute()

def _pure_permute():
    """6*6的純置換矩陣, 反轉且反向"""
    permut = np.zeros((6,6))
    for i,j in [(0,5), (1,4), (2,3), (3,2), (4,1), (5,0)]:
        permut[i,j] = -1
    return permut