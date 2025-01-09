 
    def left_leg_jacobian_wf(self):
        pelvis = copy.deepcopy(self.P_PV_wf)
        l_hip_roll = copy.deepcopy(self.P_Lhr_wf)
        l_hip_yaw = copy.deepcopy(self.P_Lhy_wf)
        l_hip_pitch = copy.deepcopy(self.P_Lhp_wf)
        l_knee_pitch = copy.deepcopy(self.P_Lkp_wf)
        l_ankle_pitch = copy.deepcopy(self.P_Lap_wf)
        l_ankle_roll = copy.deepcopy(self.P_Lar_wf)
        l_foot = copy.deepcopy(self.P_L_wf)

        JL1 = np.cross(self.AL1,(l_foot-l_hip_roll),axis=0)
        JL2 = np.cross(self.AL2,(l_foot-l_hip_yaw),axis=0)
        JL3 = np.cross(self.AL3,(l_foot-l_hip_pitch),axis=0)
        JL4 = np.cross(self.AL4,(l_foot-l_knee_pitch),axis=0)
        JL5 = np.cross(self.AL5,(l_foot-l_ankle_pitch),axis=0)
        JL6 = np.cross(self.AL6,(l_foot-l_ankle_roll),axis=0)

        JLL_upper = np.hstack((JL1, JL2,JL3,JL4,JL5,JL6))
        JLL_lower = np.hstack((self.AL1,self.AL2,self.AL3,self.AL4,self.AL5,self.AL6))    
        self.JLL = np.vstack((JLL_upper,JLL_lower))  
        # print(self.JLL)

        #排除支撐腳腳踝對末端速度的影響
        self.JL_sp44 = np.reshape(self.JLL[2:,0:4],(4,4))  
        self.JL_sp42 = np.reshape(self.JLL[2:,4:],(4,2))

        return self.JLL

    def right_leg_jacobian_wf(self):
        pelvis = copy.deepcopy(self.P_PV_wf)
        r_hip_roll = copy.deepcopy(self.P_Rhr_wf)
        r_hip_yaw = copy.deepcopy(self.P_Rhy_wf)
        r_hip_pitch = copy.deepcopy(self.P_Rhp_wf)
        r_knee_pitch = copy.deepcopy(self.P_Rkp_wf)
        r_ankle_pitch = copy.deepcopy(self.P_Rap_wf)
        r_ankle_roll = copy.deepcopy(self.P_Rar_wf)
        r_foot = copy.deepcopy(self.P_R_wf)

        JR1 = np.cross(self.AR1,(r_foot-r_hip_roll),axis=0)
        JR2 = np.cross(self.AR2,(r_foot-r_hip_yaw),axis=0)
        JR3 = np.cross(self.AR3,(r_foot-r_hip_pitch),axis=0)
        JR4 = np.cross(self.AR4,(r_foot-r_knee_pitch),axis=0)
        JR5 = np.cross(self.AR5,(r_foot-r_ankle_pitch),axis=0)
        JR6 = np.cross(self.AR6,(r_foot-r_ankle_roll),axis=0)

        JRR_upper = np.hstack((JR1,JR2,JR3,JR4,JR5,JR6))
        JRR_lower = np.hstack((self.AR1,self.AR2,self.AR3,self.AR4,self.AR5,self.AR6))    
        self.JRR = np.vstack((JRR_upper,JRR_lower))  
        # print(self.JRR)

        #排除支撐腳腳踝對末端速度的影響
        self.JR_sp44 = np.reshape(self.JRR[2:,0:4],(4,4))  
        self.JR_sp42 = np.reshape(self.JRR[2:,4:],(4,2))
        return self.JRR
     
    def alip_test(self,joint_position,joint_velocity,l_leg_vcmd,r_leg_vcmd,l_leg_gravity_compensate,r_leg_gravity_compensate,kl,kr,px_in_lf):
        print("alip_test")
        jp = copy.deepcopy(joint_position)
        jv = copy.deepcopy(joint_velocity)
        vl_cmd = copy.deepcopy(l_leg_vcmd)
        vr_cmd = copy.deepcopy(r_leg_vcmd)
        l_leg_gravity = copy.deepcopy(l_leg_gravity_compensate)
        r_leg_gravity = copy.deepcopy(r_leg_gravity_compensate)
        l_foot_in_wf = np.array([[0.0],[0.1],[0],[0],[0],[0]]) #平踏於地面時的位置
        px_in_wf = px_in_lf + l_foot_in_wf

        torque = np.zeros((12,1))

        torque[0,0] = kl*(vl_cmd[0,0]-jv[0,0]) + l_leg_gravity[0,0]
        torque[1,0] = kl*(vl_cmd[1,0]-jv[1,0]) + l_leg_gravity[1,0]
        torque[2,0] = kl*(vl_cmd[2,0]-jv[2,0]) + l_leg_gravity[2,0]
        torque[3,0] = kl*(vl_cmd[3,0]-jv[3,0]) + l_leg_gravity[3,0]
        torque[4,0] = 0
        torque[5,0] = 20*(0.2-jp[5,0]) + l_leg_gravity[5,0]

        torque[6,0] = kr*(vr_cmd[0,0]-jv[6,0]) + r_leg_gravity[0,0]
        torque[7,0] = kr*(vr_cmd[1,0]-jv[7,0])+ r_leg_gravity[1,0]
        torque[8,0] = kr*(vr_cmd[2,0]-jv[8,0]) + r_leg_gravity[2,0]
        torque[9,0] = kr*(vr_cmd[3,0]-jv[9,0]) + r_leg_gravity[3,0]
        torque[10,0] = kr*(vr_cmd[4,0]-jv[10,0]) + r_leg_gravity[4,0]
        torque[11,0] = kr*(vr_cmd[5,0]-jv[11,0]) + r_leg_gravity[5,0]

        collect_data = [str(px_in_wf[0,0])]
        csv_file_name = '/home/ldsc/com_x.csv'
        with open(csv_file_name, 'a', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)
            # Write the data
            csv_writer.writerow(collect_data)
        print(f'Data has been written to {csv_file_name}.')

        return torque
    
     
    def foot_data(self,px_in_lf,px_in_rf,L,torque_L,com_in_lf):
        l_foot_in_wf = np.array([[0.0],[0.1],[0],[0],[0],[0]]) #平踏於地面時的位置
        r_foot_in_wf = np.array([[0.007],[-0.1],[0],[0],[0],[0]]) #平踏於地面時的位置
        px_in_wf = px_in_lf + l_foot_in_wf
        # px_in_wf = px_in_lf + r_foot_in_wf
        ref_data = np.array([[self.PX_ref[0,0]],[self.PX_ref[1,0]],[self.PX_ref[2,0]],[self.PX_ref[3,0]],[self.PX_ref[4,0]],[self.PX_ref[5,0]]])
        self.ref_publisher.publish(Float64MultiArray(data=ref_data))
        
        # pelvis_data = np.array([[px_in_wf[0,0]],[px_in_wf[1,0]],[px_in_wf[2,0]],[px_in_wf[3,0]],[px_in_wf[4,0]],[px_in_wf[5,0]]])
        pelvis_data = np.array([[px_in_lf[0,0]],[px_in_lf[1,0]],[px_in_lf[2,0]],[px_in_lf[3,0]],[px_in_lf[4,0]],[px_in_lf[5,0]]])
        self.pelvis_publisher.publish(Float64MultiArray(data=pelvis_data))

        if self.state == 9: #ALIP_X實驗
            collect_data = [str(self.PX_ref[0,0]),str(self.PX_ref[1,0]),str(self.PX_ref[2,0]),str(self.PX_ref[3,0]),str(self.PX_ref[4,0]),str(self.PX_ref[5,0]),
            str(px_in_wf[0,0]),str(px_in_wf[1,0]),str(px_in_wf[2,0]),str(px_in_wf[3,0]),str(px_in_wf[4,0]),str(px_in_wf[5,0])]
            csv_file_name = '/home/ldsc/pelvis.csv'
            with open(csv_file_name, 'a', newline='') as csvfile:
                # Create a CSV writer object
                csv_writer = csv.writer(csvfile)
                # Write the data
                csv_writer.writerow(collect_data)
            print(f'Data has been written to {csv_file_name}.')

        if self.state == 7: #ALIP_L平衡實驗
            collect_data = [str(px_in_lf[0,0]),str(px_in_lf[1,0]),str(px_in_lf[2,0]),str(px_in_lf[3,0]),str(px_in_lf[4,0]),str(px_in_lf[5,0])
                            ,str(torque_L[4,0]),str(torque_L[5,0])]
            csv_file_name = '/home/ldsc/impulse_test.csv'
            with open(csv_file_name, 'a', newline='') as csvfile:
                # Create a CSV writer object
                csv_writer = csv.writer(csvfile)
                # Write the data
                csv_writer.writerow(collect_data)
            print(f'Data has been written to {csv_file_name}.')

        if self.state == 4: #ALIP_L質心軌跡追蹤實驗
            collect_data = [str(px_in_lf[2,0]),str(px_in_lf[3,0]),str(px_in_lf[4,0]),str(px_in_lf[5,0]),str(torque_L[4,0]),str(torque_L[5,0])]
            # csv_file_name = '/home/ldsc/alip_tracking_attitude.csv'
            # with open(csv_file_name, 'a', newline='') as csvfile:
            #     # Create a CSV writer object
            #     csv_writer = csv.writer(csvfile)
            #     # Write the data
            #     csv_writer.writerow(collect_data)
            # print(f'Data has been written to {csv_file_name}.')

    def to_matlab(self):
    #only x_y_z
    P_PV_wf = copy.deepcopy(self.P_PV_wf)
    P_COM_wf = copy.deepcopy(self.P_COM_wf)
    P_L_wf = copy.deepcopy(self.P_L_wf)
    P_R_wf = copy.deepcopy(self.P_R_wf)
    
    self.PX_publisher.publish(Float64MultiArray(data=P_PV_wf))
    self.COM_publisher.publish(Float64MultiArray(data=P_COM_wf))
    self.LX_publisher.publish(Float64MultiArray(data=P_L_wf))
    self.RX_publisher.publish(Float64MultiArray(data=P_R_wf))


                     
    def get_initial_data(self,stance):
        P_L_wf = copy.deepcopy(self.P_L_wf)
        P_R_wf = copy.deepcopy(self.P_R_wf)
        #怎麼順利地拿?
        #直接拿切換後的支撐腳所估測出的狀態當成初始狀態不合理 因為該估測狀態所用的扭矩來自該腳仍是擺動腳時所得到的
        #所以初始狀態選擇拿量測值(我現在的想法)

        #藉由支撐狀態切換
        if stance == 1: #(左單支撐)
            #支撐frame
            P_cf_wf = P_L_wf
            O_wfcf = np.array([[1,0,0],[0,1,0],[0,0,1]])
            #初始狀態
            X0 = copy.deepcopy(self.mea_x_L)
            Y0 = copy.deepcopy(self.mea_y_L)
            #擺動腳前一刻狀態
            O_cfwf = np.transpose(O_wfcf)
            Psw2com_X_0 = O_cfwf@(self.P_COM_wf - self.P_R_wf)
            Psw2com_0 = np.array([[Psw2com_X_0[0,0]],[Psw2com_X_0[1,0]]])
        else:#(右單支撐)
            #支撐frame
            P_cf_wf = P_R_wf
            O_wfcf = np.array([[1,0,0],[0,1,0],[0,0,1]])
            #初始狀態
            X0 = copy.deepcopy(self.mea_x_R)
            Y0 = copy.deepcopy(self.mea_y_R)
            #擺動腳前一刻狀態
            O_cfwf = np.transpose(O_wfcf)
            Psw2com_X_0 = O_cfwf@(self.P_COM_wf - self.P_L_wf)
            Psw2com_0 = np.array([[Psw2com_X_0[0,0]],[Psw2com_X_0[1,0]]])
        
        return P_cf_wf,X0,Y0,Psw2com_0

    def online_planning(self,stance,contact_t,P_cf_wf,X0,Y0,P_Psw2com_0):

        t = copy.deepcopy(contact_t)#該支撐狀態運行時間
        P_cf_wf = copy.deepcopy(P_cf_wf) #contact frame 在 wf 上的位置(xyz)
        O_wfcf = np.array([[1,0,0],[0,1,0],[0,0,1]]) #contact frame 在 wf 上的姿態
        X0 = copy.deepcopy(X0) #切換至該支撐下，初始狀態(xc(0)、ly(0))
        Y0 = copy.deepcopy(Y0) #切換至該支撐下，初始狀態(yc(0)、lx(0))
        Psw2com_x_0 = copy.deepcopy(P_Psw2com_0[0,0])
        Psw2com_y_0 = copy.deepcopy(P_Psw2com_0[1,0])
        cosh = math.cosh
        sinh = math.sinh
        cos = math.cos
        sin = math.sin
        pi = math.pi

        #理想機器人狀態
        m = 9    #機器人下肢總重
        H = 0.45 #理想質心高度
        W = 0.2 #兩腳底間距
        g = 9.81 #重力
        l = math.sqrt(g/H)


        #會用到的參數
        T = 0.5 #支撐間格時長
        Vx_des_2T = 0.15 #下兩步踩踏時刻x方向理想速度
        Ly_des_2T = m*Vx_des_2T*H #下兩步踩踏時刻相對於接觸點的理想y方向角動量
        #下兩步踩踏時刻相對於接觸點的理想x方向角動量
        Lx_des_2T_1 = (0.5*m*H*W)*(l*sinh(l*T))/(1+cosh(l*T)) #當下一次支撐腳是左腳(現在是右單支)
        Lx_des_2T_2 = -(0.5*m*H*W)*(l*sinh(l*T))/(1+cosh(l*T)) #當下一次支撐腳是右腳(現在是左單支)
        #踏步高度
        zCL = 0.02
        
        # print("t",t)
        
        #理想ALIP模型動態
        ALIP_x = np.array([[cosh(l*t),(sinh(l*t)/(m*H*l))],[m*H*l*sinh(l*t),cosh(l*t)]])
        ALIP_y = np.array([[cosh(l*t),-(sinh(l*t)/(m*H*l))],[-m*H*l*sinh(l*t),cosh(l*t)]])
        
        # print(" ALIP_x", ALIP_x)
        # print("X0",X0)

        #質心參考軌跡
        #理想上，質心相對接觸點的frame的位置及角動量(用ALIP求)，這會被拿去當成支撐腳ALIP的參考命令
        Xx_cf = np.reshape(ALIP_x@X0,(2,1))#(xc、ly)
        Xy_cf = np.reshape(ALIP_y@Y0,(2,1))#(yc、lx)

        #debug
        # print("Xx:",Xx_cf)
        
        #理想上，質心相對接觸點的位置(x,y,z)
        Com_x_cf = copy.deepcopy(Xx_cf[0,0])
        Com_y_cf = copy.deepcopy(Xy_cf[0,0])
        Com_z_cf = copy.deepcopy(H)
        #轉成大地座標下的軌跡
        Com_ref_wf = O_wfcf@np.array([[Com_x_cf],[Com_y_cf],[Com_z_cf]]) + P_cf_wf
       
        #支撐腳參考軌跡
        Support_ref_wf = P_cf_wf#延續

        #擺動腳參考軌跡
        #更新下一次踩踏瞬間時理想角動量數值
        Ly_T = m*H*l*sinh(l*T)*X0[0,0] + cosh(l*T)*X0[1,0]
        Lx_T = -m*H*l*sinh(l*T)*Y0[0,0] + cosh(l*T)*Y0[1,0]
        #根據下一次支撐腳切換lx
        if stance == 1:
            Lx_des_2T = Lx_des_2T_2
        else:
            Lx_des_2T = Lx_des_2T_1
        #理想上，下一步擺動腳踩踏點(相對於下一步踩踏時刻下的質心位置)
        Psw2com_x_T = (Ly_des_2T - cosh(l*T)*Ly_T)/(m*H*l*sinh(l*T))
        Psw2com_y_T = (Lx_des_2T - cosh(l*T)*Lx_T)/-(m*H*l*sinh(l*T))
        #理想上，擺動腳相對接觸點的位置(x,y,z)
        pv = t/T #變數 用於連接擺動腳軌跡
        Sw_x_cf = Com_x_cf - (0.5*((1+cos(pi*pv))*Psw2com_x_0 + (1-cos(pi*pv))*Psw2com_x_T))
        Sw_y_cf = Com_y_cf - (0.5*((1+cos(pi*pv))*Psw2com_y_0 + (1-cos(pi*pv))*Psw2com_y_T))
        Sw_z_cf = Com_z_cf - (4*zCL*(pv-0.5)**2 + (H-zCL))
        #轉成大地座標下的軌跡
        Swing_ref_wf = O_wfcf@np.array([[Sw_x_cf],[Sw_y_cf],[Sw_z_cf]]) + P_cf_wf

        #根據支撐狀態分配支撐腳軌跡、擺動腳軌跡、ALIP參考軌跡(質心、角動量)
        if stance == 1:
            L_ref_wf = Support_ref_wf
            R_ref_wf = Swing_ref_wf
            self.ref_x_L = copy.deepcopy(np.array([[Xx_cf[0,0]],[Xx_cf[1,0]]]))
            self.ref_y_L = copy.deepcopy(np.array([[Xy_cf[0,0]],[Xy_cf[1,0]]]))
        else:
            L_ref_wf = Swing_ref_wf
            R_ref_wf = Support_ref_wf
            self.ref_x_R = np.array([[Xx_cf[0,0]],[Xx_cf[1,0]]])
            self.ref_y_R = np.array([[Xy_cf[0,0]],[Xy_cf[1,0]]])
        
        return Com_ref_wf,L_ref_wf,R_ref_wf
    
    def gravity_ALIP(self,joint_position,stance_type,px_in_lf,px_in_rf,l_contact,r_contact):
        jp_l = np.reshape(copy.deepcopy(joint_position[0:6,0]),(6,1)) #左腳
        jp_r = np.reshape(copy.deepcopy(joint_position[6:,0]),(6,1))  #右腳
        stance = copy.deepcopy((stance_type))

        #DS_gravity
        jp_L_DS = np.flip(-jp_l,axis=0)
        jv_L_DS = np.zeros((6,1))
        c_L_DS = np.zeros((6,1))
        L_DS_gravity = np.reshape(-pin.rnea(self.stance_l_model, self.stance_l_data, jp_L_DS,jv_L_DS,(c_L_DS)),(6,1))  
        L_DS_gravity = np.flip(L_DS_gravity,axis=0)

        jp_R_DS = np.flip(-jp_r,axis=0)
        jv_R_DS = np.zeros((6,1))
        c_R_DS = np.zeros((6,1))
        R_DS_gravity = np.reshape(-pin.rnea(self.stance_r_model, self.stance_r_data, jp_R_DS,jv_R_DS,(c_R_DS)),(6,1))  
        R_DS_gravity = np.flip(R_DS_gravity,axis=0)
        DS_gravity = np.vstack((L_DS_gravity, R_DS_gravity))

        #RSS_gravity
        jp_R_RSS = np.flip(-jp_r,axis=0)
        jp_RSS = np.vstack((jp_R_RSS,jp_l))
        jv_RSS = np.zeros((12,1))
        c_RSS = np.zeros((12,1))
        Leg_RSS_gravity = np.reshape(pin.rnea(self.bipedal_r_model, self.bipedal_r_data, jp_RSS,jv_RSS,(c_RSS)),(12,1))  

        L_RSS_gravity = np.reshape(Leg_RSS_gravity[6:,0],(6,1))
        R_RSS_gravity = np.reshape(-Leg_RSS_gravity[0:6,0],(6,1)) #加負號(相對關係)
        R_RSS_gravity = np.flip(R_RSS_gravity,axis=0)
        RSS_gravity = np.vstack((L_RSS_gravity, R_RSS_gravity))

        #LSS_gravity
        jp_L_LSS = np.flip(-jp_l,axis=0)
        jp_LSS = np.vstack((jp_L_LSS,jp_r))
        jv_LSS = np.zeros((12,1))
        c_LSS = np.zeros((12,1))
        Leg_LSS_gravity = np.reshape(pin.rnea(self.bipedal_l_model, self.bipedal_l_data, jp_LSS,jv_LSS,(c_LSS)),(12,1))  

        L_LSS_gravity = np.reshape(-Leg_LSS_gravity[0:6,0],(6,1)) #加負號(相對關係)
        L_LSS_gravity = np.flip(L_LSS_gravity,axis=0)
        R_LSS_gravity = np.reshape(Leg_LSS_gravity[6:,0],(6,1))
        LSS_gravity = np.vstack((L_LSS_gravity, R_LSS_gravity))

        # if r_contact == 1:
        #     if self.stance_past == 1:
        #         kr = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
        #         kl = np.array([[1.2],[1.2],[1.2],[1.5],[1.5],[1.5]])
        #         # zero_gravity = np.zeros((6,1))
        #         # Leg_gravity = np.vstack((L_LSS_gravity, zero_gravity))
        #         Leg_gravity = 0.15*DS_gravity+0.85*RSS_gravity
        #     else:
        #         kr = np.array([[1.2],[1.2],[1.2],[1.5],[1.5],[1.5]])
        #         kl = np.array([[1],[1],[1],[1],[1],[1]])
        #         Leg_gravity = 0.15*DS_gravity+0.85*RSS_gravity
        
        # elif l_contact == 1:
        #     if self.stance_past == 0:
        #         kl = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
        #         kr = np.array([[1.2],[1.2],[1.2],[1.5],[1.5],[1.5]])
        #         # zero_gravity = np.zeros((6,1))
        #         # Leg_gravity = np.vstack((zero_gravity,R_RSS_gravity))
        #         Leg_gravity =  0.15*DS_gravity+0.85*LSS_gravity
        #     else:
        #         kl = np.array([[1.2],[1.2],[1.2],[1.5],[1.5],[1.5]])
        #         kr = np.array([[1],[1],[1],[1],[1],[1]])
        #         Leg_gravity =  0.15*DS_gravity+0.85*LSS_gravity
        # else:
        #     kr = np.array([[0.8],[0.8],[0.8],[0.8],[0],[0]])
        #     kl = np.array([[0.8],[0.8],[0.8],[0.8],[0],[0]])
        #     zero_gravity = np.zeros((6,1))
        #     Leg_gravity = np.vstack((zero_gravity, zero_gravity))

        if stance == 0:
            if r_contact == 1:
                kr = np.array([[1.2],[1.2],[1.2],[1.5],[1.5],[1.5]])
            else:
                kr = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            kl = np.array([[1],[1],[1],[1],[0],[0]])
            zero_gravity = np.zeros((12,1))
            # Leg_gravity = 0.25*zero_gravity+0.75*RSS_gravity
            Leg_gravity = 0.3*DS_gravity+0.75*RSS_gravity
        
        elif stance == 1:
            kr = np.array([[1],[1],[1],[1],[0],[0]])
            if l_contact == 1:
                kl = np.array([[1.2],[1.2],[1.2],[1.5],[1.5],[1.5]])
            else:
                kl = np.array([[1.2],[1.2],[1.2],[1.2],[1.2],[1.2]])
            zero_gravity = np.zeros((12,1))
            # Leg_gravity =  0.25*zero_gravity+0.75*LSS_gravity
            Leg_gravity =  0.3*DS_gravity+0.75*LSS_gravity

        l_leg_gravity = np.reshape(Leg_gravity[0:6,0],(6,1))
        r_leg_gravity = np.reshape(Leg_gravity[6:,0],(6,1))

        self.l_gravity_publisher.publish(Float64MultiArray(data=l_leg_gravity))
        self.r_gravity_publisher.publish(Float64MultiArray(data=r_leg_gravity))
        
        return l_leg_gravity,r_leg_gravity,kl,kr
