within ;
package ControlTests
  model RankineCycle "Adapted Rankine From Transform"
    extends TRANSFORM.Icons.Example;
    package Medium = Modelica.Media.Water.StandardWater "Working fluid";
    parameter Modelica.Units.SI.MassFlowRate m_flow=100 "Flow rate in cycle";
    parameter Modelica.Units.SI.Pressure p_steam=8.6e6 "Steam pressure";
    parameter Modelica.Units.SI.Temperature T_steam=
        Modelica.Units.Conversions.from_degC(500) "Steam temperature";
    parameter Modelica.Units.SI.Pressure p_condenser=1e4 "Condenser pressure";
    parameter Modelica.Units.SI.PressureDifference dp_pump=p_steam -
        p_condenser;
    Modelica.Mechanics.Rotational.Sensors.PowerSensor powerSensor
      annotation (Placement(transformation(extent={{24,50},{44,30}})));
    TRANSFORM.Electrical.Sources.FrequencySource boundary(f=60)
      annotation (Placement(transformation(extent={{96,30},{76,50}})));
    inner TRANSFORM.Fluid.System system
      annotation (Placement(transformation(extent={{60,80},{80,100}})));
    TRANSFORM.Fluid.Machines.SteamTurbine steamTurbine(
      m_flow_start=100,
      redeclare package Medium = Modelica.Media.Water.StandardWater,
      p_a_start=p_steam,
      p_b_start=p_condenser,
      T_a_start=T_steam)
      annotation (Placement(transformation(extent={{0,30},{20,50}})));
    TRANSFORM.Fluid.Volumes.IdealCondenser condenser(redeclare package Medium =
          Modelica.Media.Water.StandardWater, p(displayUnit="Pa") = p_condenser)
      annotation (Placement(transformation(extent={{50,-28},{70,-8}})));
    Modelica.Fluid.Vessels.ClosedVolume boiler(
      use_portsData=false,
      use_HeatTransfer=true,
      V=0.01,
      nPorts=3,
      redeclare package Medium = Modelica.Media.Water.StandardWater,
      p_start=p_steam,
      T_start=T_steam)
                      annotation (Placement(transformation(
          extent={{-10,-10},{10,10}},
          rotation=90,
          origin={-60,8})));
    TRANSFORM.Fluid.Machines.Pump pump(
      redeclare model FlowChar =
          TRANSFORM.Fluid.ClosureRelations.PumpCharacteristics.Models.Head.PerformanceCurve
          (V_flow_curve={0,0.1,2*0.1}, head_curve={1000,500,0}),
      m_flow_nominal=m_flow,
      use_T_start=false,
      m_flow_start=m_flow,
      h_start=191.8e3,
      V=0,
      redeclare model EfficiencyChar =
          TRANSFORM.Fluid.ClosureRelations.PumpCharacteristics.Models.Efficiency.Constant
          (eta_constant=0.8),
      p_a_start=p_condenser,
      controlType="pressure",
      exposeState_a=false,
      exposeState_b=true,
      dp_nominal=dp_pump,
      redeclare package Medium = Modelica.Media.Water.StandardWater)
      annotation (Placement(transformation(extent={{10,-84},{-10,-64}})));
    TRANSFORM.Electrical.PowerConverters.Generator generator
      annotation (Placement(transformation(extent={{50,30},{70,50}})));
    TRANSFORM.Fluid.Sensors.SpecificEnthalpy specificEnthalpy_out(redeclare
        package Medium = Modelica.Media.Water.StandardWater)
      annotation (Placement(transformation(extent={{-90,40},{-70,60}})));
    TRANSFORM.Fluid.Sensors.SpecificEnthalpy specificEnthalpy_in(redeclare
        package Medium =
                 Modelica.Media.Water.StandardWater)
      annotation (Placement(transformation(extent={{-90,-67},{-70,-47}})));
    TRANSFORM.Fluid.Sensors.MassFlowRate massFlowRate(redeclare package Medium =
          Modelica.Media.Water.StandardWater)        annotation (Placement(
          transformation(
          extent={{-10,-10},{10,10}},
          rotation=90,
          origin={-40,-50})));
  Modelica.Units.SI.Power Q_totalTh "Total thermal power";
  Modelica.Units.SI.Efficiency eta_overall "Overall Rankine efficiency";
    TRANSFORM.Utilities.ErrorAnalysis.UnitTests unitTests(
      n=1,
      printResult=false,
      x={pump.medium.p})
      annotation (Placement(transformation(extent={{80,80},{100,100}})));
    Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature
      prescribedTemperature
      annotation (Placement(transformation(extent={{-162,-26},{-128,8}})));
    TRANSFORM.Fluid.Valves.ValveLinear TCV(
      redeclare package Medium = Modelica.Media.Water.StandardWater,
      m_flow_start=400,
      dp_nominal=100000,
      m_flow_nominal=40) annotation (Placement(transformation(
          extent={{8,8},{-8,-8}},
          rotation=180,
          origin={-22,44})));
    Modelica.Blocks.Sources.Constant const1(k=100e6)
      annotation (Placement(transformation(extent={{-126,118},{-112,132}})));
    TRANSFORM.Controls.LimPID TCV_Power(
      controllerType=Modelica.Blocks.Types.SimpleController.PI,
      with_FF=true,
      k=1e-6,
      Ti=1,
      k_s=1,
      k_m=1,
      yMax=0,
      yMin=-1 + 0.005,
      initType=Modelica.Blocks.Types.Init.NoInit,
      xi_start=1500)
      annotation (Placement(transformation(extent={{-96,136},{-76,116}})));
    Modelica.Blocks.Sources.Constant const7(k=1)
      annotation (Placement(transformation(extent={{-74,112},{-66,120}})));
    Modelica.Blocks.Math.Add         add1
      annotation (Placement(transformation(extent={{-56,112},{-36,132}})));
    Modelica.Blocks.Sources.Ramp ramp(
      height=800,
      duration=0.01,
      offset=773.15,
      startTime=50)
      annotation (Placement(transformation(extent={{-276,-48},{-234,-6}})));
    Modelica.Blocks.Sources.Ramp ramp1(
      height=-0.4,
      duration=0.001,
      offset=0,
      startTime=50)
      annotation (Placement(transformation(extent={{-266,146},{-246,166}})));
  equation
    Q_totalTh = massFlowRate.m_flow*(specificEnthalpy_out.h_out - specificEnthalpy_in.h_out);
    eta_overall =boundary.port.W/Q_totalTh;
    connect(steamTurbine.shaft_b, powerSensor.flange_a)
      annotation (Line(points={{20,40},{20,40},{24,40}}, color={0,0,0}));
    connect(steamTurbine.portLP, condenser.port_a)
      annotation (Line(points={{20,46},{20,-11},{53,-11}},
                                                         color={0,127,255}));
    connect(powerSensor.flange_b, generator.shaft)
      annotation (Line(points={{44,40},{44,40},{50,40}}, color={0,0,0}));
    connect(generator.port, boundary.port)
      annotation (Line(points={{70,40},{76,40},{76,40}},       color={255,0,0}));
    connect(massFlowRate.port_b, boiler.ports[1]) annotation (Line(points={{-40,-40},
            {-40,-40},{-40,4},{-40,6.66667},{-50,6.66667}}, color={0,127,255}));
    connect(specificEnthalpy_out.port, boiler.ports[2])
      annotation (Line(points={{-80,40},{-50,40},{-50,8}}, color={0,127,255}));
    connect(pump.port_b, massFlowRate.port_a) annotation (Line(points={{-10,-74},{
            -40,-74},{-40,-60}}, color={0,127,255}));
    connect(pump.port_a, condenser.port_b)
      annotation (Line(points={{10,-74},{60,-74},{60,-26}}, color={0,127,255}));
    connect(specificEnthalpy_in.port, massFlowRate.port_a) annotation (Line(
          points={{-80,-67},{-80,-74},{-40,-74},{-40,-60}}, color={0,127,255}));
    connect(prescribedTemperature.port, boiler.heatPort) annotation (Line(
          points={{-128,-9},{-128,-10},{-60,-10},{-60,-2}}, color={191,0,0}));
    connect(boiler.ports[3], TCV.port_a) annotation (Line(points={{-50,9.33333},
            {-50,44},{-30,44}}, color={0,127,255}));
    connect(TCV.port_b, steamTurbine.portHP)
      annotation (Line(points={{-14,44},{-14,46},{0,46}}, color={0,127,255}));
    connect(TCV_Power.y,add1. u1)
      annotation (Line(points={{-75,126},{-68,126},{-68,128},{-58,128}},
                                                     color={0,0,127}));
    connect(const7.y,add1. u2) annotation (Line(points={{-65.6,116},{-58,116}},
                                        color={0,0,127}));
    connect(const1.y, TCV_Power.u_s) annotation (Line(points={{-111.3,125},{
            -104,125},{-104,126},{-98,126}}, color={0,0,127}));
    connect(powerSensor.power, TCV_Power.u_m) annotation (Line(points={{26,51},
            {26,150},{-86,150},{-86,138}}, color={0,0,127}));
    connect(add1.y, TCV.opening) annotation (Line(points={{-35,122},{-22,122},{
            -22,50.4}}, color={0,0,127}));
    connect(ramp.y, prescribedTemperature.T) annotation (Line(points={{-231.9,
            -27},{-231.9,-28},{-176,-28},{-176,-9},{-165.4,-9}}, color={0,0,127}));
    connect(ramp1.y, TCV_Power.u_ff) annotation (Line(points={{-245,156},{-134,
            156},{-134,112},{-106,112},{-106,118},{-98,118}}, color={0,0,127}));
    annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
          coordinateSystem(preserveAspectRatio=false)));
  end RankineCycle;

  model SteamTurbine_L2_OpenFeedHeat_Test
    import NHES;

    extends Modelica.Icons.Example;
    SteamTurbine_L2_OpenFeedHeat_NoPump                                 BOP(
      redeclare ControlTests.CS_SteamTurbine_L2_OFWH_NoPump CS(
          electric_demand_int=ramp.y, data(T_Steam_Ref=673.15, Q_Nom=40e6)),
      redeclare replaceable NHES.Systems.BalanceOfPlant.Turbine.Data.Turbine_2
        data(
        T_Steam_Ref=673.15,
        Q_Nom=40e6,
        V_tee=50,
        valve_TCV_mflow=10,
        valve_TCV_dp_nominal=20000,
        valve_TBV_mflow=4,
        valve_TBV_dp_nominal=2000000,
        InternalBypassValve_p_spring=6500000,
        InternalBypassValve_K(unit="1/(m.kg)"),
        InternalBypassValve_tau(unit="1/s"),
        MainFeedHeater_K_tube(unit="1/m4"),
        MainFeedHeater_K_shell(unit="1/m4"),
        BypassFeedHeater_K_tube(unit="1/m4"),
        BypassFeedHeater_K_shell(unit="1/m4")),
      port_a_nominal(
        m_flow=67,
        p=3400000,
        h=3e6),
      port_b_nominal(p=3500000, h=1e6),
      init(
        tee_p_start=800000,
        moisturesep_p_start=700000,
        FeedwaterMixVolume_p_start=100000,
        HPT_T_b_start=578.15,
        MainFeedHeater_p_start_shell=100000,
        MainFeedHeater_h_start_shell_inlet=2e6,
        MainFeedHeater_h_start_shell_outlet=1.8e6,
        MainFeedHeater_dp_init_shell=90000,
        MainFeedHeater_m_start_tube=67,
        MainFeedHeater_m_start_shell=67,
        BypassFeedHeater_h_start_tube_inlet=1.1e6,
        BypassFeedHeater_h_start_tube_outlet=1.2e6,
        BypassFeedHeater_m_start_tube=67,
        BypassFeedHeater_m_start_shell=4),
      deaerator(
        energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
        massDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
        p_start=200000),
      Pump_Speed(yMax=100))
      annotation (Placement(transformation(extent={{-2,-30},{58,30}})));
    TRANSFORM.Electrical.Sources.FrequencySource
                                       sinkElec(f=60)
      annotation (Placement(transformation(extent={{90,-10},{70,10}})));
    NHES.Fluid.Sensors.stateSensor stateSensor(redeclare package Medium =
          Modelica.Media.Water.StandardWater)
      annotation (Placement(transformation(extent={{-34,-24},{-54,-4}})));
    NHES.Fluid.Sensors.stateSensor stateSensor1(redeclare package Medium =
          Modelica.Media.Water.StandardWater)
      annotation (Placement(transformation(extent={{-38,2},{-18,22}})));
    NHES.Fluid.Sensors.stateDisplay stateDisplay
      annotation (Placement(transformation(extent={{-110,-46},{-66,-16}})));
    NHES.Fluid.Sensors.stateDisplay stateDisplay1
      annotation (Placement(transformation(extent={{-90,28},{-44,58}})));
    Modelica.Blocks.Sources.Sine sine(
      f=1/200,
      offset=4e8,
      startTime=350,
      amplitude=2e8)
      annotation (Placement(transformation(extent={{-70,70},{-50,90}})));
    NHES.Fluid.Pipes.StraightPipe_withWall pipe(
      redeclare package Medium =
          Modelica.Media.Water.StandardWater,
      nV=8,
      p_a_start=3400000,
      p_b_start=3500000,
      use_Ts_start=false,
      T_a_start=421.15,
      T_b_start=579.15,
      h_a_start=1.2e6,
      h_b_start=2.2e6,
      m_flow_a_start=67,
      length=10,
      diameter=1,
      redeclare package Wall_Material = NHES.Media.Solids.SS316,
      th_wall=0.001) annotation (Placement(transformation(
          extent={{-10,-10},{10,10}},
          rotation=90,
          origin={-60,0})));
    TRANSFORM.HeatAndMassTransfer.BoundaryConditions.Heat.HeatFlow boundary(use_port=
          true, Q_flow=500e6)
      annotation (Placement(transformation(extent={{-96,-10},{-76,10}})));
    Modelica.Blocks.Sources.Pulse pulse(
      amplitude=20e6,
      period=5000,
      offset=140e6,
      startTime=3000)
      annotation (Placement(transformation(extent={{-232,20},{-212,40}})));
    Modelica.Blocks.Sources.Constant const3(k=400 + 273.15)
      annotation (Placement(transformation(extent={{-220,-78},{-200,-58}})));
    TRANSFORM.Controls.LimPID Pump_Speed(
      controllerType=Modelica.Blocks.Types.SimpleController.PI,
      with_FF=true,
      k=-0.1,
      Ti=100,
      yMax=250,
      yMin=-70,
      wp=0.8,
      Ni=0.1,
      y_start=30)
      annotation (Placement(transformation(extent={{-118,-90},{-104,-76}})));
    Modelica.Blocks.Sources.Constant const4(k=70)
      annotation (Placement(transformation(extent={{-98,-72},{-90,-64}})));
    Modelica.Blocks.Math.Add         add
      annotation (Placement(transformation(extent={{-82,-84},{-62,-64}})));
    TRANSFORM.Fluid.Machines.Pump_SimpleMassFlow pump_SimpleMassFlow1(
      m_flow_nominal=70,
      use_input=true,
      redeclare package Medium = Modelica.Media.Water.StandardWater)
                                                           annotation (
        Placement(transformation(
          extent={{-11,-11},{11,11}},
          rotation=180,
          origin={-19,-13})));
    TRANSFORM.Fluid.Sensors.PressureTemperature
                                         sensor_pT(redeclare package Medium =
          Modelica.Media.Water.StandardWater, redeclare function iconUnit =
          TRANSFORM.Units.Conversions.Functions.Pressure_Pa.to_bar)
                                                         annotation (Placement(
          transformation(
          extent={{10,-10},{-10,10}},
          rotation=0,
          origin={-12,50})));
    Modelica.Blocks.Sources.Ramp ramp(
      height=-5e6,
      duration=100,
      offset=40e6,
      startTime=10000)
      annotation (Placement(transformation(extent={{0,70},{20,90}})));
    Modelica.Blocks.Sources.Constant const(k=140e6)
      annotation (Placement(transformation(extent={{-236,-18},{-216,2}})));
    Modelica.Blocks.Sources.Sine sine1
      annotation (Placement(transformation(extent={{-264,-58},{-230,-24}})));
    Modelica.Blocks.Sources.Trapezoid trapezoid(
      amplitude=1,
      rising=100,
      width=2000,
      falling=1,
      period=4000,
      offset=0,
      startTime=10000)
      annotation (Placement(transformation(extent={{-314,-40},{-294,-20}})));
  equation

    connect(stateDisplay1.statePort, stateSensor1.statePort) annotation (Line(
          points={{-67,39.1},{-68,39.1},{-68,22},{-42,22},{-42,26},{-27.95,26},
            {-27.95,12.05}},                                         color={0,0,0}));
    connect(stateDisplay.statePort, stateSensor.statePort) annotation (Line(
          points={{-88,-34.9},{-88,-50},{-44.05,-50},{-44.05,-13.95}},
                                                             color={0,0,0}));
    connect(stateSensor1.port_b, BOP.port_a)
      annotation (Line(points={{-18,12},{-2,12}},  color={0,127,255}));
    connect(BOP.portElec_b, sinkElec.port)
      annotation (Line(points={{58,0},{70,0}}, color={255,0,0}));
    connect(stateSensor.port_b, pipe.port_a) annotation (Line(points={{-54,-14},
            {-58,-14},{-58,-10},{-60,-10}},      color={0,127,255}));
    connect(pipe.port_b, stateSensor1.port_a)
      annotation (Line(points={{-60,10},{-60,12},{-38,12}}, color={0,127,255}));
    connect(boundary.port, pipe.heatPorts[1])
      annotation (Line(points={{-76,0},{-64.4,0},{-64.4,-1.25625}},
                                                               color={191,0,0}));
    connect(const3.y,Pump_Speed. u_s) annotation (Line(points={{-199,-68},{-124,
            -68},{-124,-83},{-119.4,-83}},
                                    color={0,0,127}));
    connect(add.u2,Pump_Speed. y) annotation (Line(points={{-84,-80},{-84,-88},
            {-100,-88},{-100,-83},{-103.3,-83}},
                         color={0,0,127}));
    connect(const4.y, add.u1)
      annotation (Line(points={{-89.6,-68},{-84,-68}}, color={0,0,127}));
    connect(stateSensor.port_a, pump_SimpleMassFlow1.port_b) annotation (Line(
          points={{-34,-14},{-34,-13},{-30,-13}}, color={0,127,255}));
    connect(BOP.port_b, pump_SimpleMassFlow1.port_a) annotation (Line(points={{
            -2,-12},{-5,-12},{-5,-13},{-8,-13}}, color={0,127,255}));
    connect(add.y, pump_SimpleMassFlow1.in_m_flow) annotation (Line(points={{
            -61,-74},{-19,-74},{-19,-21.03}}, color={0,0,127}));
    connect(stateSensor1.port_b, sensor_pT.port) annotation (Line(points={{-18,
            12},{-12,12},{-12,40}}, color={0,127,255}));
    connect(sensor_pT.T, Pump_Speed.u_m) annotation (Line(points={{-18,47.8},{
            -40,47.8},{-40,62},{-156,62},{-156,-91.4},{-111,-91.4}}, color={0,0,
            127}));
    connect(const.y, boundary.Q_flow_ext) annotation (Line(points={{-215,-8},{
            -102,-8},{-102,0},{-90,0}}, color={0,0,127}));
    connect(trapezoid.y, Pump_Speed.u_ff) annotation (Line(points={{-293,-30},{
            -270,-30},{-270,-86},{-126,-86},{-126,-77.4},{-119.4,-77.4}}, color=
           {0,0,127}));
    annotation (experiment(
        StopTime=10000,
        Interval=10,
        __Dymola_Algorithm="Esdirk45a"));
  end SteamTurbine_L2_OpenFeedHeat_Test;

  model SteamTurbine_L2_OpenFeedHeat_NoPump "Two stage BOP model"
    extends NHES.Systems.BalanceOfPlant.Turbine.BaseClasses.Partial_SubSystem_C(
      redeclare replaceable CS_SteamTurbine_L2_OFWH_NoPump CS,
      redeclare replaceable
        NHES.Systems.BalanceOfPlant.Turbine.ControlSystems.ED_Dummy ED,
      redeclare replaceable NHES.Systems.BalanceOfPlant.Turbine.Data.Turbine_2
        data(InternalBypassValve_p_spring=6500000));

    TRANSFORM.Fluid.Machines.SteamTurbine HPT(
      nUnits=1,
      energyDynamics=TRANSFORM.Types.Dynamics.DynamicFreeInitial,
      Q_units_start={1e7},
      eta_mech=data.HPT_efficiency,
      redeclare model Eta_wetSteam =
          TRANSFORM.Fluid.Machines.BaseClasses.WetSteamEfficiency.eta_Constant,
      p_a_start=init.HPT_p_a_start,
      p_b_start=init.HPT_p_b_start,
      T_a_start=init.HPT_T_a_start,
      T_b_start=init.HPT_T_b_start,
      m_flow_nominal=data.HPT_nominal_mflow,
      p_inlet_nominal= data.p_in_nominal,
      p_outlet_nominal=data.HPT_p_exit_nominal,
      use_T_nominal=false,
      T_nominal=data.HPT_T_in_nominal)
      annotation (Placement(transformation(extent={{32,22},{52,42}})));

    NHES.Fluid.Vessels.IdealCondenser Condenser(
      p=data.p_condensor,
      V_total=data.V_condensor,
      V_liquid_start=init.condensor_V_liquid_start)
      annotation (Placement(transformation(extent={{156,-112},{136,-92}})));

    TRANSFORM.Fluid.Sensors.TemperatureTwoPort
                                         sensor_T1(redeclare package Medium =
          Modelica.Media.Water.StandardWater)            annotation (Placement(
          transformation(
          extent={{6,6},{-6,-6}},
          rotation=180,
          origin={22,40})));

    TRANSFORM.Fluid.Sensors.PressureTemperature
                                         sensor_pT(
                                                  redeclare package Medium =
          Modelica.Media.Water.StandardWater, redeclare function iconUnit =
          TRANSFORM.Units.Conversions.Functions.Pressure_Pa.to_bar)
                                                         annotation (Placement(
          transformation(
          extent={{10,-10},{-10,10}},
          rotation=0,
          origin={-18,60})));

    TRANSFORM.Fluid.Valves.ValveLinear TCV(
      redeclare package Medium = Modelica.Media.Water.StandardWater,
      m_flow_start=400,
      dp_nominal=data.valve_TCV_dp_nominal,
      m_flow_nominal=data.valve_TCV_mflow)
                         annotation (Placement(transformation(
          extent={{8,8},{-8,-8}},
          rotation=180,
          origin={-4,40})));

    TRANSFORM.Fluid.Machines.SteamTurbine LPT(
      nUnits=1,
      energyDynamics=TRANSFORM.Types.Dynamics.DynamicFreeInitial,
      Q_units_start={3e7},
      eta_mech=data.LPT_efficiency,
      redeclare model Eta_wetSteam =
          TRANSFORM.Fluid.Machines.BaseClasses.WetSteamEfficiency.eta_Constant,
      p_a_start=init.LPT_p_a_start,
      p_b_start=init.LPT_p_b_start,
      T_a_start=init.LPT_T_a_start,
      T_b_start=init.LPT_T_b_start,
      m_flow_nominal=data.LPT_nominal_mflow,
      p_inlet_nominal= data.LPT_p_in_nominal,
      p_outlet_nominal=data.LPT_p_exit_nominal,
      T_nominal=data.LPT_T_in_nominal) annotation (Placement(transformation(
          extent={{10,10},{-10,-10}},
          rotation=90,
          origin={44,-6})));

    TRANSFORM.Fluid.FittingsAndResistances.TeeJunctionVolume tee(redeclare
        package Medium = Modelica.Media.Water.StandardWater, V=data.V_tee,
      p_start=init.tee_p_start,
      T_start=init.moisturesep_T_start)
      annotation (Placement(transformation(extent={{-10,10},{10,-10}},
          rotation=90,
          origin={82,24})));

    TRANSFORM.Fluid.Sensors.TemperatureTwoPort
                                         sensor_T2(redeclare package Medium =
          Modelica.Media.Water.StandardWater)            annotation (Placement(
          transformation(
          extent={{-10,-10},{10,10}},
          rotation=180,
          origin={-58,-40})));

    TRANSFORM.Fluid.Machines.Pump_SimpleMassFlow
                                             firstfeedpump(redeclare package
        Medium =
          Modelica.Media.Water.StandardWater,
      use_input=true,
      m_flow_nominal=25,
      allowFlowReversal=false)
      annotation (Placement(transformation(extent={{10,-10},{-10,10}},
          rotation=0,
          origin={108,-144})));

    TRANSFORM.Fluid.Volumes.MixingVolume FeedwaterMixVolume(
      redeclare package Medium = Modelica.Media.Examples.TwoPhaseWater,
      p_start=init.FeedwaterMixVolume_p_start,
      use_T_start=false,
      h_start=init.FeedwaterMixVolume_h_start,
      redeclare model Geometry =
          TRANSFORM.Fluid.ClosureRelations.Geometry.Models.LumpedVolume.GenericVolume
          (V=data.V_FeedwaterMixVolume),
      nPorts_a=2,
      nPorts_b=1) annotation (Placement(transformation(
          extent={{-10,-10},{10,10}},
          rotation=-90,
          origin={28,-54})));

    NHES.Electrical.Generator generator1(J=data.generator_MoI) annotation (
        Placement(transformation(
          extent={{-10,-10},{10,10}},
          rotation=-90,
          origin={44,-38})));

    TRANSFORM.Electrical.Sensors.PowerSensor sensorW
      annotation (Placement(transformation(extent={{110,-58},{130,-38}})));

    TRANSFORM.Fluid.FittingsAndResistances.SpecifiedResistance R_entry(R=data.R_entry,
        redeclare package Medium = Modelica.Media.Water.StandardWater)
      annotation (Placement(transformation(
          extent={{10,-10},{-10,10}},
          rotation=180,
          origin={-132,40})));

    TRANSFORM.Fluid.Volumes.MixingVolume header(
      use_T_start=false,
      h_start=init.header_h_start,
      p_start=init.header_p_start,
      nPorts_a=1,
      nPorts_b=1,
      redeclare model Geometry =
          TRANSFORM.Fluid.ClosureRelations.Geometry.Models.LumpedVolume.GenericVolume
          (V=1),
      redeclare package Medium = Modelica.Media.Water.StandardWater)
      annotation (Placement(transformation(extent={{-122,30},{-102,50}})));

    TRANSFORM.Fluid.BoundaryConditions.Boundary_pT boundary(
      redeclare package Medium = Modelica.Media.Water.StandardWater,
      p=data.p_boundary,
      T=data.T_boundary,
      nPorts=1)
      annotation (Placement(transformation(extent={{-168,64},{-148,84}})));

    TRANSFORM.Fluid.Valves.ValveLinear TBV(
      redeclare package Medium = Modelica.Media.Water.StandardWater,
      dp_nominal=data.valve_TBV_dp_nominal,
      m_flow_nominal=data.valve_TBV_mflow) annotation (Placement(transformation(
          extent={{-8,8},{8,-8}},
          rotation=180,
          origin={-128,74})));

    TRANSFORM.Fluid.Sensors.TemperatureTwoPort
                                         sensor_T4(redeclare package Medium =
          Modelica.Media.Water.StandardWater)            annotation (Placement(
          transformation(
          extent={{10,-10},{-10,10}},
          rotation=180,
          origin={80,-144})));

    replaceable NHES.Systems.BalanceOfPlant.Turbine.Data.Turbine_2_init init(
        FeedwaterMixVolume_h_start=2e6)
      annotation (Placement(transformation(extent={{68,120},{88,140}})));

    TRANSFORM.Fluid.Volumes.Deaerator deaerator(
      redeclare model Geometry =
          TRANSFORM.Fluid.ClosureRelations.Geometry.Models.TwoVolume_withLevel.Cylinder
          (
          V_liquid=10,
          length=5,
          r_inner=2,
          th_wall=0.1),
      redeclare package Medium = Modelica.Media.Water.StandardWater,
      level_start=4,
      p_start=300000,
      use_T_start=false,
      d_wall=1000,
      cp_wall=420,
      Twall_start=373.15)
      annotation (Placement(transformation(extent={{56,-122},{36,-102}})));
    TRANSFORM.Fluid.FittingsAndResistances.SpecifiedResistance R_entry1(R=data.R_entry,
        redeclare package Medium = Modelica.Media.Water.StandardWater)
      annotation (Placement(transformation(
          extent={{10,-10},{-10,10}},
          rotation=90,
          origin={30,-80})));
    Modelica.Blocks.Sources.RealExpression FWTank_level(y=deaerator.level)
      "level"
      annotation (Placement(transformation(extent={{180,-138},{192,-126}})));
    Modelica.Blocks.Sources.Constant const1(k=3)
      annotation (Placement(transformation(extent={{178,-108},{192,-94}})));
    TRANSFORM.Controls.LimPID Pump_Speed(
      controllerType=Modelica.Blocks.Types.SimpleController.PI,
      with_FF=false,
      k=30,
      Ti=500,
      yb=0.01,
      k_s=0.9,
      k_m=0.9,
      yMax=40,
      yMin=2,
      wp=1,
      Ni=0.001,
      xi_start=0,
      y_start=0.01)
      annotation (Placement(transformation(extent={{220,-120},{234,-106}})));
    NHES.Systems.BalanceOfPlant.StagebyStageTurbineSecondary.StagebyStageTurbine.BaseClasses.TRANSFORMMoistureSeparator_MIKK
      Moisture_Separator(
      redeclare package Medium = Modelica.Media.Water.StandardWater,
      p_start=init.moisturesep_p_start,
      T_start=init.moisturesep_T_start,
      redeclare model Geometry =
          TRANSFORM.Fluid.ClosureRelations.Geometry.Models.LumpedVolume.GenericVolume
          (V=data.V_moistureseperator))
      annotation (Placement(transformation(extent={{58,24},{78,44}})));
  initial equation

  equation

    connect(HPT.portHP, sensor_T1.port_b) annotation (Line(
        points={{32,38},{30,38},{30,40},{28,40}},
        color={0,127,255},
        thickness=0.5));
    connect(TCV.port_b, sensor_T1.port_a) annotation (Line(
        points={{4,40},{16,40}},
        color={0,127,255},
        thickness=0.5));
    connect(LPT.portHP, tee.port_1) annotation (Line(
        points={{50,4},{50,8},{82,8},{82,14}},
        color={0,127,255},
        thickness=0.5));
    connect(sensorBus.Feedwater_Temp, sensor_T2.T) annotation (Line(
        points={{-30,100},{-44,100},{-44,-56},{-58,-56},{-58,-43.6}},
        color={239,82,82},
        pattern=LinePattern.Dash,
        thickness=0.5));
    connect(HPT.shaft_b, LPT.shaft_a) annotation (Line(
        points={{52,32},{52,14},{44,14},{44,4}},
        color={0,0,0},
        pattern=LinePattern.Dash));

    connect(actuatorBus.opening_TCV, TCV.opening) annotation (Line(
        points={{30.1,100.1},{-4,100.1},{-4,46.4}},
        color={111,216,99},
        pattern=LinePattern.Dash,
        thickness=0.5));
    connect(sensor_pT.port, TCV.port_a)
      annotation (Line(points={{-18,50},{-18,40},{-12,40}}, color={0,127,255}));

    connect(LPT.shaft_b, generator1.shaft_a)
      annotation (Line(points={{44,-16},{44,-28}}, color={0,0,0}));
    connect(generator1.portElec, sensorW.port_a) annotation (Line(points={{44,-48},
            {110,-48}},                              color={255,0,0}));
    connect(sensorW.port_b, portElec_b) annotation (Line(points={{130,-48},{146,
            -48},{146,0},{160,0}},                     color={255,0,0}));
    connect(sensorBus.Steam_Pressure, sensor_pT.p) annotation (Line(
        points={{-30,100},{-30,62.4},{-24,62.4}},
        color={239,82,82},
        pattern=LinePattern.Dash,
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(sensorBus.Power, sensorW.W) annotation (Line(
        points={{-30,100},{120,100},{120,-37}},
        color={239,82,82},
        pattern=LinePattern.Dash,
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-3,6},{-3,6}},
        horizontalAlignment=TextAlignment.Right));
    connect(port_a, R_entry.port_a)
      annotation (Line(points={{-160,40},{-139,40}}, color={0,127,255}));
    connect(R_entry.port_b, header.port_a[1])
      annotation (Line(points={{-125,40},{-118,40}}, color={0,127,255}));
    connect(header.port_b[1], TCV.port_a)
      annotation (Line(points={{-106,40},{-60,40},{-60,40},{-12,40}},
                                                    color={0,127,255}));
    connect(TBV.port_a, TCV.port_a) annotation (Line(points={{-120,74},{-104,74},
            {-104,40},{-12,40}}, color={0,127,255}));
    connect(boundary.ports[1], TBV.port_b)
      annotation (Line(points={{-148,74},{-136,74}}, color={0,127,255}));
    connect(actuatorBus.TBV, TBV.opening) annotation (Line(
        points={{30,100},{-128,100},{-128,80.4}},
        color={111,216,99},
        pattern=LinePattern.Dash,
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-3,6},{-3,6}},
        horizontalAlignment=TextAlignment.Right));
    connect(firstfeedpump.port_b, sensor_T4.port_b) annotation (Line(points={{98,-144},
            {90,-144}},                              color={0,127,255}));
    connect(Condenser.port_b, firstfeedpump.port_a) annotation (Line(points={{146,
            -112},{146,-144},{118,-144}},         color={0,127,255}));
    connect(LPT.portLP, Condenser.port_a) annotation (Line(points={{50,-16},{60,
            -16},{60,-64},{154,-64},{154,-86},{153,-86},{153,-92}},
                                                           color={0,127,255}));
    connect(tee.port_3, FeedwaterMixVolume.port_a[1]) annotation (Line(points={{92,24},
            {94,24},{94,12},{27.75,12},{27.75,-48}},
          color={0,127,255}));
    connect(deaerator.feed, sensor_T4.port_a) annotation (Line(points={{53,-105},
            {64,-105},{64,-144},{70,-144}}, color={0,127,255}));
    connect(deaerator.steam, R_entry1.port_b) annotation (Line(points={{39,-105},
            {39,-96},{46,-96},{46,-87},{30,-87}}, color={0,127,255}));
    connect(R_entry1.port_a, FeedwaterMixVolume.port_b[1])
      annotation (Line(points={{30,-73},{30,-60},{28,-60}}, color={0,127,255}));
    connect(FWTank_level.y,Pump_Speed. u_m)
      annotation (Line(points={{192.6,-132},{227,-132},{227,-121.4}},
                                                             color={0,0,127}));
    connect(const1.y,Pump_Speed. u_s) annotation (Line(points={{192.7,-101},{
            192.7,-102},{214,-102},{214,-113},{218.6,-113}},
                                color={0,0,127}));
    connect(Pump_Speed.y, firstfeedpump.in_m_flow) annotation (Line(points={{
            234.7,-113},{234.7,-136.7},{108,-136.7}}, color={0,0,127}));
    connect(HPT.portLP, Moisture_Separator.port_a)
      annotation (Line(points={{52,38},{52,34},{62,34}}, color={0,127,255}));
    connect(Moisture_Separator.port_b, tee.port_2)
      annotation (Line(points={{74,34},{82,34}}, color={0,127,255}));
    connect(Moisture_Separator.port_Liquid, FeedwaterMixVolume.port_a[2])
      annotation (Line(points={{64,30},{60,30},{60,12},{28.25,12},{28.25,-48}},
          color={0,127,255}));
    connect(sensorBus.Steam_Temperature, sensor_pT.T) annotation (Line(
        points={{-30,100},{-30,57.8},{-24,57.8}},
        color={239,82,82},
        pattern=LinePattern.Dash,
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(deaerator.drain, sensor_T2.port_a) annotation (Line(points={{46,-120},
            {46,-128},{-42,-128},{-42,-40},{-48,-40}}, color={0,127,255}));
    connect(sensor_T2.port_b, port_b)
      annotation (Line(points={{-68,-40},{-160,-40}}, color={0,127,255}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false), graphics={
          Rectangle(
            extent={{-11.5,3},{11.5,-3}},
            lineColor={0,0,0},
            fillColor={64,164,200},
            fillPattern=FillPattern.HorizontalCylinder,
            origin={-1,-28.5},
            rotation=90),
          Rectangle(
            extent={{-4.5,2.5},{4.5,-2.5}},
            lineColor={0,0,0},
            fillColor={64,164,200},
            fillPattern=FillPattern.HorizontalCylinder,
            origin={-8.5,-31.5},
            rotation=360),
          Rectangle(
            extent={{-18,3},{18,-3}},
            lineColor={0,0,0},
            fillColor={66,200,200},
            fillPattern=FillPattern.HorizontalCylinder,
            origin={-39,28},
            rotation=-90),
          Rectangle(
            extent={{-1.81332,3},{66.1869,-3}},
            lineColor={0,0,0},
            origin={-18.1867,-3},
            rotation=0,
            fillColor={135,135,135},
            fillPattern=FillPattern.HorizontalCylinder),
          Rectangle(
            extent={{-70,46},{-36,34}},
            lineColor={0,0,0},
            fillColor={66,200,200},
            fillPattern=FillPattern.HorizontalCylinder),
          Polygon(
            points={{-42,12},{-42,-18},{-12,-36},{-12,32},{-42,12}},
            lineColor={0,0,0},
            fillColor={0,114,208},
            fillPattern=FillPattern.Solid),
          Text(
            extent={{-31,-10},{-21,4}},
            lineColor={0,0,0},
            fillColor={255,255,255},
            fillPattern=FillPattern.Solid,
            textString="HPT"),
          Ellipse(
            extent={{46,12},{74,-14}},
            lineColor={0,0,0},
            fillColor={255,255,255},
            fillPattern=FillPattern.Solid),
          Rectangle(
            extent={{-0.601938,3},{23.3253,-3}},
            lineColor={0,0,0},
            origin={22.6019,-29},
            rotation=0,
            fillColor={0,128,255},
            fillPattern=FillPattern.HorizontalCylinder),
          Rectangle(
            extent={{-0.43805,2.7864},{15.9886,-2.7864}},
            lineColor={0,0,0},
            origin={45.2136,-41.989},
            rotation=90,
            fillColor={0,128,255},
            fillPattern=FillPattern.HorizontalCylinder),
          Ellipse(
            extent={{32,-40},{60,-66}},
            lineColor={0,0,0},
            fillColor={255,255,255},
            fillPattern=FillPattern.Solid),
          Rectangle(
            extent={{-0.373344,2},{13.6267,-2}},
            lineColor={0,0,0},
            origin={18.3733,-54},
            rotation=0,
            fillColor={0,0,255},
            fillPattern=FillPattern.HorizontalCylinder),
          Rectangle(
            extent={{-0.341463,2},{13.6587,-2}},
            lineColor={0,0,0},
            origin={20,-42.3415},
            rotation=-90,
            fillColor={0,0,255},
            fillPattern=FillPattern.HorizontalCylinder),
          Rectangle(
            extent={{-1.41463,2.0001},{56.5851,-2.0001}},
            lineColor={0,0,0},
            origin={-25.4149,-62.0001},
            rotation=180,
            fillColor={0,0,255},
            fillPattern=FillPattern.HorizontalCylinder),
          Ellipse(
            extent={{-26,-56},{-14,-68}},
            lineColor={0,0,0},
            fillPattern=FillPattern.Sphere,
            fillColor={0,100,199}),
          Polygon(
            points={{-24,-66},{-28,-70},{-12,-70},{-16,-66},{-24,-66}},
            lineColor={0,0,255},
            pattern=LinePattern.None,
            fillColor={0,0,0},
            fillPattern=FillPattern.VerticalCylinder),
          Ellipse(
            extent={{-56,49},{-38,31}},
            lineColor={95,95,95},
            fillColor={175,175,175},
            fillPattern=FillPattern.Sphere),
          Rectangle(
            extent={{-46,49},{-48,61}},
            lineColor={0,0,0},
            fillColor={95,95,95},
            fillPattern=FillPattern.VerticalCylinder),
          Rectangle(
            extent={{-56,63},{-38,61}},
            lineColor={0,0,0},
            fillColor={181,0,0},
            fillPattern=FillPattern.HorizontalCylinder),
          Ellipse(
            extent={{-45,49},{-49,31}},
            lineColor={0,0,0},
            fillPattern=FillPattern.VerticalCylinder,
            fillColor={162,162,0}),
          Text(
            extent={{55,-10},{65,4}},
            lineColor={0,0,0},
            fillColor={255,255,255},
            fillPattern=FillPattern.Solid,
            textString="G"),
          Text(
            extent={{41,-60},{51,-46}},
            lineColor={0,0,0},
            fillColor={255,255,255},
            fillPattern=FillPattern.Solid,
            textString="C"),
          Polygon(
            points={{-19,-59},{-19,-65},{-23,-62},{-19,-59}},
            lineColor={0,0,0},
            pattern=LinePattern.None,
            fillPattern=FillPattern.HorizontalCylinder,
            fillColor={255,255,255}),
          Polygon(
            points={{-4,12},{-4,-18},{26,-36},{26,32},{-4,12}},
            lineColor={0,0,0},
            fillColor={0,114,208},
            fillPattern=FillPattern.Solid),
          Text(
            extent={{7,-10},{17,4}},
            lineColor={0,0,0},
            fillColor={255,255,255},
            fillPattern=FillPattern.Solid,
            textString="LPT"),
          Rectangle(
            extent={{-14,-40},{12,-50}},
            lineColor={28,108,200},
            fillColor={28,108,200},
            fillPattern=FillPattern.Solid),
          Ellipse(
            extent={{-18,-40},{-8,-50}},
            lineColor={28,108,200},
            fillColor={28,108,200},
            fillPattern=FillPattern.Solid),
          Ellipse(
            extent={{6,-40},{16,-50}},
            lineColor={28,108,200},
            fillColor={28,108,200},
            fillPattern=FillPattern.Solid),
          Ellipse(
            extent={{-6,-36},{4,-46}},
            lineColor={28,108,200},
            fillColor={28,108,200},
            fillPattern=FillPattern.Solid),
          Rectangle(
            extent={{-0.213341,2},{7.7867,-2}},
            lineColor={0,0,0},
            origin={14.2133,-44},
            rotation=0,
            fillColor={0,0,255},
            fillPattern=FillPattern.HorizontalCylinder),
          Rectangle(
            extent={{-0.341463,2},{13.6587,-2}},
            lineColor={0,0,0},
            origin={-2,-62.3415},
            rotation=90,
            fillColor={0,0,255},
            fillPattern=FillPattern.HorizontalCylinder),
          Rectangle(
            extent={{-0.373344,2},{13.6267,-2}},
            lineColor={0,0,0},
            origin={-1.6267,-62},
            rotation=180,
            fillColor={0,0,255},
            fillPattern=FillPattern.HorizontalCylinder)}),         Diagram(
          coordinateSystem(preserveAspectRatio=false)),
      experiment(
        StopTime=1000,
        Interval=10,
        __Dymola_Algorithm="Esdirk45a"),
      Documentation(info="<html>
<p>A two stage turbine rankine cycle with feedwater heating internal to the system - can be externally bypassed or LPT can be bypassed both will feedwater heat post bypass</p>
<p>&nbsp; </p>
<p align=\"center\"><img src=\"file:///C:/Users/RIGBAC/AppData/Local/Temp/1/msohtmlclip1/01/clip_image002.jpg\"/> </p>
<p><b><span style=\"font-size: 18pt;\">Design Purpose</span></b> </p>
<p>The main purpose of this model is to provide a simple and flexible two stage BOP with realistic accounting of feedwater heating. It should be used in cases where a more rigorous accounting of efficiency is required compared to the SteamTurbines_L1_boundaries model and the StageByStage turbine model would add unnecessary complexity. </p>
<p><b><span style=\"font-size: 18pt;\">Location and Examples</span></b> </p>
<p>The location of this model is at NHES.Systems.BalanceOfPlant.Turbine.SteamTurbine_L2_ClosedFeedHeat the best use case example of this model is at NHES.Systems.Examples.SMR_highfidelity_L2_Turbine. </p>
<p>&nbsp; </p>
<p><b><span style=\"font-size: 18pt;\">Normal Operation</span></b> </p>
<p>The model uses two TRANSFORM SteamTurbine models with the intermediate pressure to be chosen by the user (nominally set at 7 Bar). Any liquid condensing in the turbines is removed via a moisture separator. The model has closed feedwater heating with steam bled from between the two turbines fed to the main NTU heat exchanger with contact to the main feedwater flow. Additional feedwater heating can be provided with an internal bypass loop from the main steam line to a supplementary NTU heat exchanger with this flow controlled by a set pressure spring valve. This steam is used again in the main NTU heat exchanger after mixing in the feedwater mixing volume. The model uses an ideal condenser with a fixed pressure that must be specified by the user (nominally set to 0.1 Bar). In the feedwater line &ndash; fixed &ldquo;pressure booster&rdquo; pumps are used to move the steam away from saturation conditions. Depending on the set pressure between turbines these pumps must be set sufficiently to prevent saturation in either of the heat exchangers tube sides. An additional final feedwater pump is used to control pressure exiting the primary heat system. Finally, the model also has a blow-off valve to an external boundary condition on the main steam line to prevent over-pressurization. </p>
<p><b><span style=\"font-size: 18pt;\">Control system</span></b> </p>
<table cellspacing=\"0\" cellpadding=\"0\" border=\"1\" width=\"662\"><tr>
<td valign=\"top\"><p align=\"center\"><span style=\"font-size: 11pt;\">Label</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">Name</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">Controlling</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">Nominal Setpoint</span> </p></td>
</tr>
<tr>
<td valign=\"top\"><p align=\"center\"><span style=\"font-size: 11pt;\">1</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">Turbine Control Valve</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">Power (HPT and LPT)</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">40 MW</span> </p></td>
</tr>
<tr>
<td valign=\"top\"><p align=\"center\"><span style=\"font-size: 11pt;\">2</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">Low Pressure Turbine Bypass</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">Feedwater Temperature</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">148&deg;C</span> </p></td>
</tr>
<tr>
<td valign=\"top\"><p align=\"center\"><span style=\"font-size: 11pt;\">3</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">Internal Bypass Valve</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">Bypass Mass Flow</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">0 kg/s</span> </p></td>
</tr>
<tr>
<td valign=\"top\"><p align=\"center\"><span style=\"font-size: 11pt;\">4</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">Feedwater Pump</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">Steam Inlet Pressure (HPT)</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">34 bar</span> </p></td>
</tr>
<tr>
<td valign=\"top\"><p align=\"center\"><span style=\"font-size: 11pt;\">5</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">Pressure Relief Valve</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">Pressure Overloads</span> </p></td>
<td valign=\"top\"><p><span style=\"font-size: 11pt;\">150 bar</span> </p></td>
</tr>
</table>
<p><br><img src=\"file:///C:/Users/RIGBAC/AppData/Local/Temp/1/msohtmlclip1/01/clip_image004.png\"/> </p>
<p>&nbsp; </p>
<p>The control system is designed to ensure nominal conditions in normal operation. In load follow or extreme transients additional control elements may be required in the model. The three key required setpoint conditions are power, feedwater temperature and steam inlet pressure to the BOP. These are specified in the data table in the control system model. The internal bypass valve spring pressure is not a controlled variable and is set in the BOP model data table. Depending on the K value of this valve (also specified in the BOP data table) one can control the leakage mass flow required for the supplementary heat exchanger to prevent no flow errors. </p>
<p><b><span style=\"font-size: 18pt;\">Changing Parameters</span></b> </p>
<p>All parameters in the model should be accessible and changed in the data table data. All initialization conditions should be changed using the init table. These have initial value in to guide your choices or aid simulation set up. </p>
<p><b><span style=\"font-size: 18pt;\">Considerations In Parameters</span></b> </p>
<p>The key considerations when changing the turbine parameters to match an arbitrary Rankine cycle are the pressures in the fixed pressure booster pumps. These should be adjusted so the outlets of the HX tube sides are pushed away from saturation conditions. The further these exit flows are away from the saturation condition the better reliability in transient operation the model will have but this will impact your efficiencies. These pump pressures are a function of setting the intermediate pressure and the first feed pump should always be sufficiently low pressure rise for heat to flow from the bypass stream to the feedwater heat not the other way round. </p>
<p>Other considerations when parameterizing the model are listed below </p>
<p>1.<span style=\"font-family: Times New Roman; font-size: 7pt;\">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span>Valve sizes </p>
<p>a.<span style=\"font-family: Times New Roman; font-size: 7pt;\">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span>Internal Bypass Valve K value should be low enough to allow a nominal flow through the supplementary HX. </p>
<p>b.<span style=\"font-family: Times New Roman; font-size: 7pt;\">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span>Nominal conditions on TCV and LPT_Bypass should be tuned to allow the full range of desired operating conditions </p>
<p>c.<span style=\"font-family: Times New Roman; font-size: 7pt;\">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span>TBV should be set that it only opens in extreme circumstances </p>
<p>2.<span style=\"font-family: Times New Roman; font-size: 7pt;\">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span>Turbine nominal conditions </p>
<p>a.<span style=\"font-family: Times New Roman; font-size: 7pt;\">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span>These must be fine tuned to desired power output for given steam conditions. There doesn&rsquo;t seem to be an exact way to do this but it would be good to know if found. </p>
<p>3.<span style=\"font-family: Times New Roman; font-size: 7pt;\">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span>Volumes in system </p>
<p>a.<span style=\"font-family: Times New Roman; font-size: 7pt;\">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span>Many of the volumes are set too large to aid initialization. These may be changed to reflect actual BOP designs, but initialization may be more difficult. </p>
<p><b><span style=\"font-size: 18pt;\">Contact Deatils</span></b> </p>
<p>This model was designed by Aidan Rigby (<a href=\"mailto:aidan.rigby@inl.gov\">aidan.rigby@inl.gov</a> , <a href=\"mailto:acrigby@wisc.edu\">acrigby@wisc.edu</a> ). All initial questions should be directed to Daniel Mikkelson (<a href=\"mailto:Daniel.Mikkelson@inl.gov\">Daniel.Mikkelson@inl.gov</a>). </p>
</html>"));
  end SteamTurbine_L2_OpenFeedHeat_NoPump;

  model CS_SteamTurbine_L2_OFWH_NoPump
    extends NHES.Systems.BalanceOfPlant.Turbine.BaseClasses.Partial_ControlSystem;

    extends NHES.Icons.DummyIcon;

    input Real electric_demand_int = data.Q_Nom
    annotation(Dialog(tab="General"));

    TRANSFORM.Controls.LimPID Turb_Divert_Valve(
      controllerType=Modelica.Blocks.Types.SimpleController.PI,
      k=5e-3,
      Ti=5,
      Td=0.1,
      yMax=1,
      yMin=0.05,
      initType=Modelica.Blocks.Types.Init.NoInit,
      xi_start=1500)
      annotation (Placement(transformation(extent={{-58,-58},{-38,-38}})));
    Modelica.Blocks.Sources.Constant const5(k=data.T_Feedwater)
      annotation (Placement(transformation(extent={{-92,-56},{-72,-36}})));
    TRANSFORM.Controls.LimPID TCV_Power(
      controllerType=Modelica.Blocks.Types.SimpleController.PI,
      k=5e-6,
      Ti=300,
      k_s=1,
      k_m=1,
      yMax=0,
      yMin=-1 + 0.005,
      initType=Modelica.Blocks.Types.Init.NoInit,
      xi_start=1500)
      annotation (Placement(transformation(extent={{-50,-2},{-30,-22}})));
    Modelica.Blocks.Sources.RealExpression
                                     realExpression(y=electric_demand_int)
      annotation (Placement(transformation(extent={{-94,-6},{-80,6}})));
    Modelica.Blocks.Sources.Constant const7(k=1)
      annotation (Placement(transformation(extent={{-26,-28},{-18,-20}})));
    Modelica.Blocks.Math.Add         add1
      annotation (Placement(transformation(extent={{-8,-28},{12,-8}})));
    Modelica.Blocks.Sources.Constant const8(k=0)
      annotation (Placement(transformation(extent={{-32,-56},{-24,-48}})));
    Modelica.Blocks.Math.Add         add2
      annotation (Placement(transformation(extent={{-8,-56},{12,-36}})));
    NHES.Systems.BalanceOfPlant.StagebyStageTurbineSecondary.Control_and_Distribution.Timer
      timer(Start_Time=1e-2)
      annotation (Placement(transformation(extent={{-32,-44},{-24,-36}})));
    replaceable NHES.Systems.BalanceOfPlant.Turbine.Data.Turbine_2_Setpoints data(
      p_steam=3500000,
      p_steam_vent=15000000,
      T_Steam_Ref=579.75,
      Q_Nom=40e6,
      T_Feedwater=421.15)
      annotation (Placement(transformation(extent={{-98,12},{-78,32}})));
    Modelica.Blocks.Sources.Constant const3(k=400 + 273.15)
      annotation (Placement(transformation(extent={{-74,38},{-54,58}})));
    Modelica.Blocks.Sources.Constant const4(k=70)
      annotation (Placement(transformation(extent={{-20,50},{-12,58}})));
    Modelica.Blocks.Math.Add         add
      annotation (Placement(transformation(extent={{-4,38},{16,58}})));
    Modelica.Blocks.Sources.Constant const2(k=1)
      annotation (Placement(transformation(extent={{2,74},{22,94}})));
    TRANSFORM.Controls.LimPID PI_TBV(
      controllerType=Modelica.Blocks.Types.SimpleController.PI,
      k=-5e-7,
      Ti=15,
      yMax=1.0,
      yMin=0.0,
      initType=Modelica.Blocks.Types.Init.NoInit)
      annotation (Placement(transformation(extent={{-38,72},{-18,92}})));
    Modelica.Blocks.Sources.Constant const9(k=data.p_steam_vent)
      annotation (Placement(transformation(extent={{-78,72},{-58,92}})));
    TRANSFORM.Controls.LimPID Pump_Speed(
      controllerType=Modelica.Blocks.Types.SimpleController.PI,
      k=-0.1,
      Ti=100,
      yMax=250,
      yMin=-70,
      wp=0.8,
      Ni=0.1,
      y_start=30)
      annotation (Placement(transformation(extent={{-40,32},{-26,46}})));
  equation
    connect(const5.y,Turb_Divert_Valve. u_s)
      annotation (Line(points={{-71,-46},{-66,-46},{-66,-48},{-60,-48}},
                                                       color={0,0,127}));
    connect(sensorBus.Feedwater_Temp,Turb_Divert_Valve. u_m) annotation (Line(
        points={{-30,-100},{-48,-100},{-48,-60}},
        color={239,82,82},
        pattern=LinePattern.Dash,
        thickness=0.5));
    connect(sensorBus.Power, TCV_Power.u_m) annotation (Line(
        points={{-30,-100},{-100,-100},{-100,8},{-40,8},{-40,0}},
        color={239,82,82},
        pattern=LinePattern.Dash,
        thickness=0.5));
    connect(const7.y,add1. u2) annotation (Line(points={{-17.6,-24},{-10,-24}},
                                        color={0,0,127}));
    connect(TCV_Power.y, add1.u1)
      annotation (Line(points={{-29,-12},{-10,-12}}, color={0,0,127}));
    connect(add2.u2,const8. y) annotation (Line(points={{-10,-52},{-23.6,-52}},
                                                                           color=
            {0,0,127}));
    connect(add2.u1,timer. y) annotation (Line(points={{-10,-40},{-23.44,-40}},
                                                                  color={0,0,127}));
    connect(Turb_Divert_Valve.y,timer. u) annotation (Line(points={{-37,-48},{-36,
            -48},{-36,-40},{-32.8,-40}},                               color={0,0,
            127}));
    connect(actuatorBus.Divert_Valve_Position, add2.y) annotation (Line(
        points={{30,-100},{30,-46},{13,-46}},
        color={111,216,99},
        pattern=LinePattern.Dash,
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{6,3},{6,3}},
        horizontalAlignment=TextAlignment.Left));
    connect(actuatorBus.opening_BV, const2.y) annotation (Line(
        points={{30.1,-99.9},{30.1,84},{23,84}},
        color={111,216,99},
        pattern=LinePattern.Dash,
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{6,3},{6,3}},
        horizontalAlignment=TextAlignment.Left));
    connect(const4.y, add.u1)
      annotation (Line(points={{-11.6,54},{-6,54}},
                                                  color={0,0,127}));
    connect(const9.y, PI_TBV.u_s)
      annotation (Line(points={{-57,82},{-40,82}}, color={0,0,127}));
    connect(sensorBus.Steam_Pressure, PI_TBV.u_m) annotation (Line(
        points={{-30,-100},{-100,-100},{-100,70},{-28,70}},
        color={239,82,82},
        pattern=LinePattern.Dash,
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-3,-6},{-3,-6}},
        horizontalAlignment=TextAlignment.Right));
    connect(actuatorBus.TBV, PI_TBV.y) annotation (Line(
        points={{30,-100},{30,66},{-10,66},{-10,82},{-17,82}},
        color={111,216,99},
        pattern=LinePattern.Dash,
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{6,3},{6,3}},
        horizontalAlignment=TextAlignment.Left));
    connect(const3.y, Pump_Speed.u_s) annotation (Line(points={{-53,48},{-46,48},{
            -46,39},{-41.4,39}},    color={0,0,127}));
    connect(sensorBus.Steam_Temperature, Pump_Speed.u_m) annotation (Line(
        points={{-30,-100},{-100,-100},{-100,8},{-33,8},{-33,30.6}},
        color={239,82,82},
        pattern=LinePattern.Dash,
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(actuatorBus.Feed_Pump_Speed, add.y) annotation (Line(
        points={{30,-100},{30,48},{17,48}},
        color={111,216,99},
        pattern=LinePattern.Dash,
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{6,3},{6,3}},
        horizontalAlignment=TextAlignment.Left));
    connect(add.u2, Pump_Speed.y) annotation (Line(points={{-6,42},{-10,42},{-10,39},
            {-25.3,39}}, color={0,0,127}));
    connect(add1.y, actuatorBus.opening_TCV) annotation (Line(points={{13,-18},{
            30.1,-18},{30.1,-99.9}}, color={0,0,127}), Text(
        string="%second",
        index=1,
        extent={{6,3},{6,3}},
        horizontalAlignment=TextAlignment.Left));
    connect(realExpression.y, TCV_Power.u_s) annotation (Line(points={{-79.3,0},{-60,
            0},{-60,-12},{-52,-12}}, color={0,0,127}));
  end CS_SteamTurbine_L2_OFWH_NoPump;

  model SteamTurbine_L2_OpenFeedHeat_Test2
    import NHES;

    extends Modelica.Icons.Example;
    SteamTurbine_L2_OpenFeedHeat_NoPump                                 BOP(
      redeclare ControlTests.CS_SteamTurbine_L2_OFWH_NoPump CS(
          electric_demand_int=ramp.y, data(T_Steam_Ref=673.15, Q_Nom=40e6)),
      redeclare replaceable NHES.Systems.BalanceOfPlant.Turbine.Data.Turbine_2
        data(
        T_Steam_Ref=673.15,
        Q_Nom=40e6,
        V_tee=50,
        valve_TCV_mflow=10,
        valve_TCV_dp_nominal=20000,
        valve_TBV_mflow=4,
        valve_TBV_dp_nominal=2000000,
        InternalBypassValve_p_spring=6500000,
        InternalBypassValve_K(unit="1/(m.kg)"),
        InternalBypassValve_tau(unit="1/s"),
        MainFeedHeater_K_tube(unit="1/m4"),
        MainFeedHeater_K_shell(unit="1/m4"),
        BypassFeedHeater_K_tube(unit="1/m4"),
        BypassFeedHeater_K_shell(unit="1/m4")),
      port_a_nominal(
        m_flow=67,
        p=3400000,
        h=3e6),
      port_b_nominal(p=3500000, h=1e6),
      init(
        tee_p_start=800000,
        moisturesep_p_start=700000,
        FeedwaterMixVolume_p_start=100000,
        HPT_T_b_start=578.15,
        MainFeedHeater_p_start_shell=100000,
        MainFeedHeater_h_start_shell_inlet=2e6,
        MainFeedHeater_h_start_shell_outlet=1.8e6,
        MainFeedHeater_dp_init_shell=90000,
        MainFeedHeater_m_start_tube=67,
        MainFeedHeater_m_start_shell=67,
        BypassFeedHeater_h_start_tube_inlet=1.1e6,
        BypassFeedHeater_h_start_tube_outlet=1.2e6,
        BypassFeedHeater_m_start_tube=67,
        BypassFeedHeater_m_start_shell=4),
      deaerator(
        energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
        massDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
        p_start=200000),
      Pump_Speed(yMax=100))
      annotation (Placement(transformation(extent={{-2,-30},{58,30}})));
    TRANSFORM.Electrical.Sources.FrequencySource
                                       sinkElec(f=60)
      annotation (Placement(transformation(extent={{90,-10},{70,10}})));
    NHES.Fluid.Sensors.stateSensor stateSensor(redeclare package Medium =
          Modelica.Media.Water.StandardWater)
      annotation (Placement(transformation(extent={{-34,-24},{-54,-4}})));
    NHES.Fluid.Sensors.stateSensor stateSensor1(redeclare package Medium =
          Modelica.Media.Water.StandardWater)
      annotation (Placement(transformation(extent={{-38,2},{-18,22}})));
    NHES.Fluid.Sensors.stateDisplay stateDisplay
      annotation (Placement(transformation(extent={{-110,-46},{-66,-16}})));
    NHES.Fluid.Sensors.stateDisplay stateDisplay1
      annotation (Placement(transformation(extent={{-90,28},{-44,58}})));
    Modelica.Blocks.Sources.Sine sine(
      f=1/200,
      offset=4e8,
      startTime=350,
      amplitude=2e8)
      annotation (Placement(transformation(extent={{-70,70},{-50,90}})));
    NHES.Fluid.Pipes.StraightPipe_withWall pipe(
      redeclare package Medium =
          Modelica.Media.Water.StandardWater,
      nV=8,
      p_a_start=3400000,
      p_b_start=3500000,
      use_Ts_start=false,
      T_a_start=421.15,
      T_b_start=579.15,
      h_a_start=1.2e6,
      h_b_start=2.2e6,
      m_flow_a_start=67,
      length=10,
      diameter=1,
      redeclare package Wall_Material = NHES.Media.Solids.SS316,
      th_wall=0.001) annotation (Placement(transformation(
          extent={{-10,-10},{10,10}},
          rotation=90,
          origin={-60,0})));
    TRANSFORM.HeatAndMassTransfer.BoundaryConditions.Heat.HeatFlow boundary(use_port=
          true, Q_flow=500e6)
      annotation (Placement(transformation(extent={{-96,-10},{-76,10}})));
    Modelica.Blocks.Sources.Constant const3(k=400 + 273.15)
      annotation (Placement(transformation(extent={{-220,-78},{-200,-58}})));
    TRANSFORM.Controls.LimPID Pump_Speed(
      controllerType=Modelica.Blocks.Types.SimpleController.PI,
      with_FF=true,
      k=-0.01,
      Ti=100,
      yMax=250,
      yMin=-70,
      wp=0.8,
      Ni=0.1,
      y_start=30)
      annotation (Placement(transformation(extent={{-118,-90},{-104,-76}})));
    Modelica.Blocks.Sources.Constant const4(k=70)
      annotation (Placement(transformation(extent={{-98,-72},{-90,-64}})));
    Modelica.Blocks.Math.Add         add
      annotation (Placement(transformation(extent={{-82,-84},{-62,-64}})));
    TRANSFORM.Fluid.Machines.Pump_SimpleMassFlow pump_SimpleMassFlow1(
      m_flow_nominal=70,
      use_input=true,
      redeclare package Medium = Modelica.Media.Water.StandardWater)
                                                           annotation (
        Placement(transformation(
          extent={{-11,-11},{11,11}},
          rotation=180,
          origin={-19,-13})));
    TRANSFORM.Fluid.Sensors.PressureTemperature
                                         sensor_pT(redeclare package Medium =
          Modelica.Media.Water.StandardWater, redeclare function iconUnit =
          TRANSFORM.Units.Conversions.Functions.Pressure_Pa.to_bar)
                                                         annotation (Placement(
          transformation(
          extent={{10,-10},{-10,10}},
          rotation=0,
          origin={-12,50})));
    Modelica.Blocks.Sources.Ramp ramp(
      height=-10e6,
      duration=100,
      offset=40e6,
      startTime=10000)
      annotation (Placement(transformation(extent={{0,70},{20,90}})));
    Modelica.Blocks.Sources.Constant const(k=140e6)
      annotation (Placement(transformation(extent={{-234,-18},{-214,2}})));
    Modelica.Blocks.Sources.Constant FeedForward(k=0)
      annotation (Placement(transformation(extent={{-198,-46},{-178,-26}})));
  equation

    connect(stateDisplay1.statePort, stateSensor1.statePort) annotation (Line(
          points={{-67,39.1},{-68,39.1},{-68,22},{-42,22},{-42,26},{-27.95,26},
            {-27.95,12.05}},                                         color={0,0,0}));
    connect(stateDisplay.statePort, stateSensor.statePort) annotation (Line(
          points={{-88,-34.9},{-88,-50},{-44.05,-50},{-44.05,-13.95}},
                                                             color={0,0,0}));
    connect(stateSensor1.port_b, BOP.port_a)
      annotation (Line(points={{-18,12},{-2,12}},  color={0,127,255}));
    connect(BOP.portElec_b, sinkElec.port)
      annotation (Line(points={{58,0},{70,0}}, color={255,0,0}));
    connect(stateSensor.port_b, pipe.port_a) annotation (Line(points={{-54,-14},
            {-58,-14},{-58,-10},{-60,-10}},      color={0,127,255}));
    connect(pipe.port_b, stateSensor1.port_a)
      annotation (Line(points={{-60,10},{-60,12},{-38,12}}, color={0,127,255}));
    connect(boundary.port, pipe.heatPorts[1])
      annotation (Line(points={{-76,0},{-64.4,0},{-64.4,-1.25625}},
                                                               color={191,0,0}));
    connect(const3.y,Pump_Speed. u_s) annotation (Line(points={{-199,-68},{-124,
            -68},{-124,-83},{-119.4,-83}},
                                    color={0,0,127}));
    connect(add.u2,Pump_Speed. y) annotation (Line(points={{-84,-80},{-84,-88},
            {-100,-88},{-100,-83},{-103.3,-83}},
                         color={0,0,127}));
    connect(const4.y, add.u1)
      annotation (Line(points={{-89.6,-68},{-84,-68}}, color={0,0,127}));
    connect(stateSensor.port_a, pump_SimpleMassFlow1.port_b) annotation (Line(
          points={{-34,-14},{-34,-13},{-30,-13}}, color={0,127,255}));
    connect(BOP.port_b, pump_SimpleMassFlow1.port_a) annotation (Line(points={{
            -2,-12},{-5,-12},{-5,-13},{-8,-13}}, color={0,127,255}));
    connect(add.y, pump_SimpleMassFlow1.in_m_flow) annotation (Line(points={{
            -61,-74},{-19,-74},{-19,-21.03}}, color={0,0,127}));
    connect(stateSensor1.port_b, sensor_pT.port) annotation (Line(points={{-18,
            12},{-12,12},{-12,40}}, color={0,127,255}));
    connect(sensor_pT.T, Pump_Speed.u_m) annotation (Line(points={{-18,47.8},{
            -40,47.8},{-40,62},{-156,62},{-156,-91.4},{-111,-91.4}}, color={0,0,
            127}));
    connect(const.y, boundary.Q_flow_ext) annotation (Line(points={{-213,-8},{
            -102,-8},{-102,0},{-90,0}}, color={0,0,127}));
    connect(FeedForward.y, Pump_Speed.u_ff) annotation (Line(points={{-177,-36},
            {-154,-36},{-154,-77.4},{-119.4,-77.4}}, color={0,0,127}));
    annotation (experiment(
        StopTime=20000,
        Interval=1,
        __Dymola_Algorithm="Esdirk45a"));
  end SteamTurbine_L2_OpenFeedHeat_Test2;

  model SteamTurbine_L2_OpenFeedHeat_Test3
    import NHES;

    extends Modelica.Icons.Example;
    SteamTurbine_L2_OpenFeedHeat_NoPump                                 BOP(
      redeclare ControlTests.CS_SteamTurbine_L2_OFWH_NoPump CS(
          electric_demand_int=ramp.y, data(T_Steam_Ref=673.15, Q_Nom=40e6)),
      redeclare replaceable NHES.Systems.BalanceOfPlant.Turbine.Data.Turbine_2
        data(
        T_Steam_Ref=673.15,
        Q_Nom=40e6,
        V_tee=50,
        valve_TCV_mflow=10,
        valve_TCV_dp_nominal=20000,
        valve_TBV_mflow=4,
        valve_TBV_dp_nominal=2000000,
        InternalBypassValve_p_spring=6500000,
        InternalBypassValve_K(unit="1/(m.kg)"),
        InternalBypassValve_tau(unit="1/s"),
        MainFeedHeater_K_tube(unit="1/m4"),
        MainFeedHeater_K_shell(unit="1/m4"),
        BypassFeedHeater_K_tube(unit="1/m4"),
        BypassFeedHeater_K_shell(unit="1/m4")),
      port_a_nominal(
        m_flow=67,
        p=3400000,
        h=3e6),
      port_b_nominal(p=3500000, h=1e6),
      init(
        tee_p_start=800000,
        moisturesep_p_start=700000,
        FeedwaterMixVolume_p_start=100000,
        HPT_T_b_start=578.15,
        MainFeedHeater_p_start_shell=100000,
        MainFeedHeater_h_start_shell_inlet=2e6,
        MainFeedHeater_h_start_shell_outlet=1.8e6,
        MainFeedHeater_dp_init_shell=90000,
        MainFeedHeater_m_start_tube=67,
        MainFeedHeater_m_start_shell=67,
        BypassFeedHeater_h_start_tube_inlet=1.1e6,
        BypassFeedHeater_h_start_tube_outlet=1.2e6,
        BypassFeedHeater_m_start_tube=67,
        BypassFeedHeater_m_start_shell=4),
      deaerator(
        energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
        massDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
        p_start=200000),
      Pump_Speed(yMax=100))
      annotation (Placement(transformation(extent={{-2,-30},{58,30}})));
    TRANSFORM.Electrical.Sources.FrequencySource
                                       sinkElec(f=60)
      annotation (Placement(transformation(extent={{90,-10},{70,10}})));
    NHES.Fluid.Sensors.stateSensor stateSensor(redeclare package Medium =
          Modelica.Media.Water.StandardWater)
      annotation (Placement(transformation(extent={{-34,-24},{-54,-4}})));
    NHES.Fluid.Sensors.stateSensor stateSensor1(redeclare package Medium =
          Modelica.Media.Water.StandardWater)
      annotation (Placement(transformation(extent={{-38,2},{-18,22}})));
    NHES.Fluid.Sensors.stateDisplay stateDisplay
      annotation (Placement(transformation(extent={{-110,-46},{-66,-16}})));
    NHES.Fluid.Sensors.stateDisplay stateDisplay1
      annotation (Placement(transformation(extent={{-90,28},{-44,58}})));
    Modelica.Blocks.Sources.Sine sine(
      f=1/200,
      offset=4e8,
      startTime=350,
      amplitude=2e8)
      annotation (Placement(transformation(extent={{-70,70},{-50,90}})));
    NHES.Fluid.Pipes.StraightPipe_withWall pipe(
      redeclare package Medium =
          Modelica.Media.Water.StandardWater,
      nV=8,
      p_a_start=3400000,
      p_b_start=3500000,
      use_Ts_start=false,
      T_a_start=421.15,
      T_b_start=579.15,
      h_a_start=1.2e6,
      h_b_start=2.2e6,
      m_flow_a_start=67,
      length=10,
      diameter=1,
      redeclare package Wall_Material = NHES.Media.Solids.SS316,
      th_wall=0.001) annotation (Placement(transformation(
          extent={{-10,-10},{10,10}},
          rotation=90,
          origin={-60,0})));
    TRANSFORM.HeatAndMassTransfer.BoundaryConditions.Heat.HeatFlow boundary(use_port=
          true, Q_flow=500e6)
      annotation (Placement(transformation(extent={{-96,-10},{-76,10}})));
    Modelica.Blocks.Sources.Constant const3(k=400 + 273.15)
      annotation (Placement(transformation(extent={{-220,-78},{-200,-58}})));
    TRANSFORM.Controls.LimPID Pump_Speed(
      controllerType=Modelica.Blocks.Types.SimpleController.PI,
      with_FF=true,
      k=-0.01,
      Ti=100,
      yMax=250,
      yMin=-70,
      wp=0.8,
      Ni=0.1,
      y_start=30)
      annotation (Placement(transformation(extent={{-118,-90},{-104,-76}})));
    Modelica.Blocks.Sources.Constant const4(k=70)
      annotation (Placement(transformation(extent={{-98,-72},{-90,-64}})));
    Modelica.Blocks.Math.Add         add
      annotation (Placement(transformation(extent={{-82,-84},{-62,-64}})));
    TRANSFORM.Fluid.Machines.Pump_SimpleMassFlow pump_SimpleMassFlow1(
      m_flow_nominal=70,
      use_input=true,
      redeclare package Medium = Modelica.Media.Water.StandardWater)
                                                           annotation (
        Placement(transformation(
          extent={{-11,-11},{11,11}},
          rotation=180,
          origin={-19,-13})));
    TRANSFORM.Fluid.Sensors.PressureTemperature
                                         sensor_pT(redeclare package Medium =
          Modelica.Media.Water.StandardWater, redeclare function iconUnit =
          TRANSFORM.Units.Conversions.Functions.Pressure_Pa.to_bar)
                                                         annotation (Placement(
          transformation(
          extent={{10,-10},{-10,10}},
          rotation=0,
          origin={-12,50})));
    Modelica.Blocks.Sources.Ramp ramp(
      height=-10e6,
      duration=100,
      offset=40e6,
      startTime=10000)
      annotation (Placement(transformation(extent={{0,70},{20,90}})));
    Modelica.Blocks.Sources.Constant const(k=140e6)
      annotation (Placement(transformation(extent={{-234,-18},{-214,2}})));
    Modelica.Blocks.Sources.CombiTimeTable
                                      FeedForward(
      tableOnFile=true,                           table=[0,0; 9900,0; 9903,-0.03;
          9906,6.71e-10; 9909,6.71e-10; 9912,0.030000001; 9915,0.030000001;
          9918,0.030000001; 9921,0.030000001; 9924,0.030000001; 9927,
          0.030000001; 9930,0.030000001; 9933,0.030000001; 9936,0.030000001;
          9939,0.030000001; 9942,0.030000001; 9945,0.030000001; 9948,
          0.030000001; 9951,0.030000001; 9954,0.030000001; 9957,0.030000001;
          9960,0.030000001; 9963,0.030000001; 9966,0.030000001; 9969,
          0.030000001; 9972,0.030000001; 9975,0.030000001; 9978,0.030000001;
          9981,0.030000001; 9984,0.030000001; 9987,0.030000001; 9990,
          0.030000001; 9993,0.030000001; 9996,0.030000001; 9999,0.030000001;
          10002,0.030000001; 10005,0.030000001; 10008,0.060000002; 10011,0.31;
          10014,0.56; 10017,0.81; 10020,1.06; 10023,0.80999994; 10026,
          0.55999994; 10029,0.53; 10032,0.49999997; 10035,0.46999997; 10038,
          0.43999997; 10041,0.40999997; 10044,0.37999997; 10047,0.37999997;
          10050,0.34999996; 10053,0.34999996; 10056,0.31999996; 10059,
          0.31999996; 10062,0.28999996; 10065,0.25999996; 10068,0.22999996;
          10071,0.19999996; 10074,0.16999996; 10077,0.13999996; 10080,
          0.109999955; 10083,0.07999995; 10086,0.049999952; 10089,0.019999953;
          10092,-0.23000005; 10095,-0.23000005; 10098,-0.26000005; 10101,-0.26000005;
          10104,-0.01000005; 10107,-0.26000005; 10110,-0.51000005; 10113,-0.51000005;
          10116,-0.51000005; 10119,-0.51000005; 10122,-0.51000005; 10125,-0.51000005;
          10128,-0.51000005; 10131,-0.51000005; 10134,-0.51000005; 10137,-0.51000005;
          10140,-0.51000005; 10143,-0.51000005; 10146,-0.51000005; 10149,-0.51000005;
          10152,-0.51000005; 10155,-0.51000005; 10158,-0.51000005; 10161,-0.51000005;
          10164,-0.51000005; 10167,-0.51000005; 10170,-0.51000005; 10173,-0.51000005;
          10176,-0.51000005; 10179,-0.51000005; 10182,-0.51000005; 10185,-0.51000005;
          10188,-0.51000005; 10191,-0.51000005; 10194,-0.51000005; 10197,-0.51000005;
          10200,-0.51000005; 10203,-0.51000005; 10206,-0.51000005; 10209,-0.51000005;
          10212,-0.51000005; 10215,-0.51000005; 10218,-0.51000005; 10221,-0.51000005;
          10224,-0.51000005; 10227,-0.51000005; 10230,-0.51000005; 10233,-0.51000005;
          10236,-0.51000005; 10239,-0.51000005; 10242,-0.51000005; 10245,-0.51000005;
          10248,-0.51000005; 10251,-0.51000005; 10254,-0.51000005; 10257,-0.51000005;
          10260,-0.51000005; 10263,-0.51000005; 10266,-0.51000005; 10269,-0.51000005;
          10272,-0.51000005; 10275,-0.51000005; 10278,-0.51000005; 10281,-0.51000005;
          10284,-0.51000005; 10287,-0.51000005; 10290,-0.51000005; 10293,-0.51000005;
          10296,-0.51000005; 10299,-0.51000005; 10302,-0.51000005; 10305,-0.51000005;
          10308,-0.51000005; 10311,-0.51000005; 10314,-0.51000005; 10317,-0.51000005;
          10320,-0.51000005; 10323,-0.51000005; 10326,-0.51000005; 10329,-0.51000005;
          10332,-0.51000005; 10335,-0.51000005; 10338,-0.51000005; 10341,-0.51000005;
          10344,-0.51000005; 10347,-0.51000005; 10350,-0.51000005; 10353,-0.51000005;
          10356,-0.51000005; 10359,-0.51000005; 10362,-0.51000005; 10365,-0.51000005;
          10368,-0.51000005; 10371,-0.51000005; 10374,-0.51000005; 10377,-0.51000005;
          10380,-0.51000005; 10383,-0.51000005; 10386,-0.51000005; 10389,-0.51000005;
          10392,-0.51000005; 10395,-0.51000005; 10398,-0.51000005; 10401,-0.51000005;
          10404,-0.51000005; 10407,-0.51000005; 10410,-0.51000005; 10413,-0.51000005;
          10416,-0.51000005; 10419,-0.51000005; 10422,-0.51000005; 10425,-0.51000005;
          10428,-0.51000005; 10431,-0.51000005; 10434,-0.51000005; 10437,-0.51000005;
          10440,-0.51000005; 10443,-0.51000005; 10446,-0.51000005; 10449,-0.51000005;
          10452,-0.51000005; 10455,-0.51000005; 10458,-0.51000005; 10461,-0.51000005;
          10464,-0.51000005; 10467,-0.51000005; 10470,-0.51000005; 10473,-0.51000005;
          10476,-0.51000005; 10479,-0.51000005; 10482,-0.51000005; 10485,-0.51000005;
          10488,-0.51000005; 10491,-0.51000005; 10494,-0.51000005; 10497,-0.51000005;
          10500,-0.51000005; 10503,-0.51000005; 10506,-0.51000005; 10509,-0.51000005;
          10512,-0.51000005; 10515,-0.51000005; 10518,-0.51000005; 10521,-0.51000005;
          10524,-0.51000005; 10527,-0.51000005; 10530,-0.51000005; 10533,-0.51000005;
          10536,-0.51000005; 10539,-0.51000005; 10542,-0.51000005; 10545,-0.51000005;
          10548,-0.51000005; 10551,-0.51000005; 10554,-0.51000005; 10557,-0.51000005;
          10560,-0.51000005; 10563,-0.51000005; 10566,-0.51000005; 10569,-0.51000005;
          10572,-0.51000005; 10575,-0.51000005; 10578,-0.51000005; 10581,-0.51000005;
          10584,-0.51000005; 10587,-0.51000005; 10590,-0.51000005; 10593,-0.51000005;
          10596,-0.51000005; 10599,-0.51000005; 10602,-0.51000005; 10605,-0.51000005;
          10608,-0.51000005; 10611,-0.51000005; 10614,-0.51000005; 10617,-0.51000005;
          10620,-0.51000005; 10623,-0.51000005; 10626,-0.51000005; 10629,-0.51000005;
          10632,-0.51000005; 10635,-0.51000005; 10638,-0.51000005; 10641,-0.51000005;
          10644,-0.51000005; 10647,-0.51000005; 10650,-0.51000005; 10653,-0.51000005;
          10656,-0.51000005; 10659,-0.51000005; 10662,-0.51000005; 10665,-0.51000005;
          10668,-0.51000005; 10671,-0.51000005; 10674,-0.51000005; 10677,-0.51000005;
          10680,-0.51000005; 10683,-0.51000005; 10686,-0.51000005; 10689,-0.51000005;
          10692,-0.51000005; 10695,-0.51000005; 10698,-0.51000005; 10701,-0.51000005;
          10704,-0.51000005; 10707,-0.51000005; 10710,-0.51000005; 10713,-0.51000005;
          10716,-0.51000005; 10719,-0.51000005; 10722,-0.51000005; 10725,-0.51000005;
          10728,-0.51000005; 10731,-0.51000005; 10734,-0.51000005; 10737,-0.51000005;
          10740,-0.51000005; 10743,-0.51000005; 10746,-0.51000005; 10749,-0.51000005;
          10752,-0.51000005; 10755,-0.51000005; 10758,-0.51000005; 10761,-0.51000005;
          10764,-0.51000005; 10767,-0.51000005; 10770,-0.51000005; 10773,-0.51000005;
          10776,-0.51000005; 10779,-0.51000005; 10782,-0.51000005; 10785,-0.51000005;
          10788,-0.51000005; 10791,-0.51000005; 10794,-0.51000005; 10797,-0.51000005;
          10800,-0.51000005; 10803,-0.51000005; 10806,-0.51000005; 10809,-0.51000005;
          10812,-0.51000005; 10815,-0.51000005; 10818,-0.51000005; 10821,-0.51000005;
          10824,-0.51000005; 10827,-0.51000005; 10830,-0.51000005; 10833,-0.51000005;
          10836,-0.51000005; 10839,-0.51000005; 10842,-0.51000005; 10845,-0.51000005;
          10848,-0.51000005; 10851,-0.51000005; 10854,-0.51000005; 10857,-0.51000005;
          10860,-0.51000005; 10863,-0.51000005; 10866,-0.51000005; 10869,-0.51000005;
          10872,-0.51000005; 10875,-0.51000005; 10878,-0.51000005; 10881,-0.51000005;
          10884,-0.51000005; 10887,-0.51000005; 10890,-0.51000005; 10893,-0.51000005;
          10896,-0.51000005; 10899,-0.51000005; 10902,-0.51000005; 10905,-0.51000005;
          10908,-0.51000005; 10911,-0.51000005; 10914,-0.51000005; 10917,-0.51000005;
          10920,-0.51000005; 10923,-0.51000005; 10926,-0.51000005; 10929,-0.51000005;
          10932,-0.51000005; 10935,-0.51000005; 10938,-0.51000005; 10941,-0.51000005;
          10944,-0.51000005; 10947,-0.51000005; 10950,-0.51000005; 10953,-0.51000005;
          10956,-0.51000005; 10959,-0.51000005; 10962,-0.51000005; 10965,-0.51000005;
          10968,-0.51000005; 10971,-0.51000005; 10974,-0.51000005; 10977,-0.51000005;
          10980,-0.51000005; 10983,-0.51000005; 10986,-0.51000005; 10989,-0.51000005;
          10992,-0.51000005; 10995,-0.51000005; 10998,-0.51000005; 11001,-0.51000005;
          11004,-0.51000005; 11007,-0.51000005; 11010,-0.51000005; 11013,-0.51000005;
          11016,-0.51000005; 11019,-0.51000005; 11022,-0.51000005; 11025,-0.51000005;
          11028,-0.51000005; 11031,-0.51000005; 11034,-0.51000005; 11037,-0.51000005;
          11040,-0.51000005; 11043,-0.51000005; 11046,-0.51000005; 11049,-0.51000005;
          11052,-0.51000005; 11055,-0.51000005; 11058,-0.51000005; 11061,-0.51000005;
          11064,-0.51000005; 11067,-0.51000005; 11070,-0.51000005; 11073,-0.51000005;
          11076,-0.51000005; 11079,-0.51000005; 11082,-0.51000005; 11085,-0.51000005;
          11088,-0.51000005; 11091,-0.51000005; 11094,-0.51000005; 11097,-0.51000005;
          11100,-0.51000005; 11103,-0.51000005; 11106,-0.51000005; 11109,-0.51000005;
          11112,-0.51000005; 11115,-0.51000005; 11118,-0.51000005; 11121,-0.51000005;
          11124,-0.51000005; 11127,-0.51000005; 11130,-0.51000005; 20000,-0.51],
      tableName="data",
      fileName="feedforward.mat")
      annotation (Placement(transformation(extent={{-198,-46},{-178,-26}})));
  equation

    connect(stateDisplay1.statePort, stateSensor1.statePort) annotation (Line(
          points={{-67,39.1},{-68,39.1},{-68,22},{-42,22},{-42,26},{-27.95,26},
            {-27.95,12.05}},                                         color={0,0,0}));
    connect(stateDisplay.statePort, stateSensor.statePort) annotation (Line(
          points={{-88,-34.9},{-88,-50},{-44.05,-50},{-44.05,-13.95}},
                                                             color={0,0,0}));
    connect(stateSensor1.port_b, BOP.port_a)
      annotation (Line(points={{-18,12},{-2,12}},  color={0,127,255}));
    connect(BOP.portElec_b, sinkElec.port)
      annotation (Line(points={{58,0},{70,0}}, color={255,0,0}));
    connect(stateSensor.port_b, pipe.port_a) annotation (Line(points={{-54,-14},
            {-58,-14},{-58,-10},{-60,-10}},      color={0,127,255}));
    connect(pipe.port_b, stateSensor1.port_a)
      annotation (Line(points={{-60,10},{-60,12},{-38,12}}, color={0,127,255}));
    connect(boundary.port, pipe.heatPorts[1])
      annotation (Line(points={{-76,0},{-64.4,0},{-64.4,-1.25625}},
                                                               color={191,0,0}));
    connect(const3.y,Pump_Speed. u_s) annotation (Line(points={{-199,-68},{-124,
            -68},{-124,-83},{-119.4,-83}},
                                    color={0,0,127}));
    connect(add.u2,Pump_Speed. y) annotation (Line(points={{-84,-80},{-84,-88},
            {-100,-88},{-100,-83},{-103.3,-83}},
                         color={0,0,127}));
    connect(const4.y, add.u1)
      annotation (Line(points={{-89.6,-68},{-84,-68}}, color={0,0,127}));
    connect(stateSensor.port_a, pump_SimpleMassFlow1.port_b) annotation (Line(
          points={{-34,-14},{-34,-13},{-30,-13}}, color={0,127,255}));
    connect(BOP.port_b, pump_SimpleMassFlow1.port_a) annotation (Line(points={{
            -2,-12},{-5,-12},{-5,-13},{-8,-13}}, color={0,127,255}));
    connect(add.y, pump_SimpleMassFlow1.in_m_flow) annotation (Line(points={{
            -61,-74},{-19,-74},{-19,-21.03}}, color={0,0,127}));
    connect(stateSensor1.port_b, sensor_pT.port) annotation (Line(points={{-18,
            12},{-12,12},{-12,40}}, color={0,127,255}));
    connect(sensor_pT.T, Pump_Speed.u_m) annotation (Line(points={{-18,47.8},{
            -40,47.8},{-40,62},{-156,62},{-156,-91.4},{-111,-91.4}}, color={0,0,
            127}));
    connect(const.y, boundary.Q_flow_ext) annotation (Line(points={{-213,-8},{
            -102,-8},{-102,0},{-90,0}}, color={0,0,127}));
    connect(Pump_Speed.u_ff, FeedForward.y[1]) annotation (Line(points={{-119.4,
            -77.4},{-119.4,-76},{-118,-76},{-118,-36},{-177,-36}}, color={0,0,
            127}));
    annotation (experiment(
        StopTime=20000,
        Interval=1,
        __Dymola_Algorithm="Esdirk45a"));
  end SteamTurbine_L2_OpenFeedHeat_Test3;
  annotation (uses(
      TRANSFORM(version="0.5"),
      Modelica(version="4.0.0"),
      NHES(version="3")));
end ControlTests;
