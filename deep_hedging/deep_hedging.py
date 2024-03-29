from tensorflow.keras.layers import Input, Dense, Concatenate, Subtract, \
                Lambda, Add, Dot, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal, Zeros, he_uniform, TruncatedNormal
import tensorflow.keras.backend as Kbackend
import tensorflow as tf
import numpy as np

intitalizer_dict = { 
    "he_normal": he_normal(),
    "zeros": Zeros(),
    "he_uniform": he_uniform(),
    "truncated_normal": TruncatedNormal()
}

bias_initializer=he_uniform()

class Strategy_Layer(tf.keras.layers.Layer):
    def __init__(self, d = None, m = None, num_instr = None, is_barrier = None, \
        use_batch_norm = None, kernel_initializer = "he_uniform", \
        activation_dense = "relu", activation_output = "linear", 
        delta_constraint = None, day = None):
        super().__init__(name = "delta_" + str(day))
        self.d = d #number of layers
        self.m = m #number of nodes
        self.num_instr = num_instr #number of hedging instruments
        self.is_barrier = is_barrier #boolean: barrier or not
        self.use_batch_norm = use_batch_norm #boolean: use batch normalization or not
        self.activation_dense = activation_dense #activation function for hidden layers
        self.activation_output = activation_output #activation fucntion for output layer
        self.delta_constraint = delta_constraint #hedging strategy constraint
        self.kernel_initializer = kernel_initializer #kernel initializer
        
        self.intermediate_dense = [None for _ in range(d)]
        self.intermediate_BN = [None for _ in range(d)]
        
        for i in range(d):
            self.intermediate_dense[i] = Dense(self.m,
                                               kernel_initializer=self.kernel_initializer,
                                               bias_initializer=bias_initializer,
                                               use_bias = (not self.use_batch_norm))
            if self.use_batch_norm:
                self.intermediate_BN[i] = BatchNormalization(momentum = 0.99, trainable=True)
           
        self.output_dense = Dense(self.num_instr, 
                      kernel_initializer=self.kernel_initializer,
                      bias_initializer = bias_initializer,
                      use_bias=True)     
        
    def call(self, input):
        for i in range(self.d):
            if self.is_barrier:
                input1 = input[0]
                input2 = input[1]
                if i == 0:
                    output = self.intermediate_dense[i](input1)
                    output = Concatenate()([output,input2])
                else:
                    output = self.intermediate_dense[i](output)    
            else:
                if i == 0:    
                    output = self.intermediate_dense[i](input)
                else:
                    output = self.intermediate_dense[i](output)                
                
            if self.use_batch_norm:
 			      # Batch normalization.
                output = self.intermediate_BN[i](output, training=True)
                
            if self.activation_dense == "leaky_relu":
                output = LeakyReLU()(output)
            else:
                output = Activation(self.activation_dense)(output)
         
        output = self.output_dense(output)
					 
        if self.activation_output == "leaky_relu":
            output = LeakyReLU()(output)
        elif self.activation_output == "sigmoid" or self.activation_output == "tanh" or \
            self.activation_output == "hard_sigmoid":
            # Enforcing hedge constraints
            if self.delta_constraint is not None:
                output = Activation(self.activation_output)(output)
                delta_min, delta_max = self.delta_constraint
                output = Lambda(lambda x : (delta_max-delta_min)*x + delta_min)(output)
            else:
                output = Activation(self.activation_output)(output)
        
        return output
    
def Deep_Hedging_Model(N = None, d = None, m = None, num_instr = None, is_barrier = None,\
        risk_free = None, dt = None, initial_wealth = 0.0, epsilon = 0.0, \
        final_period_cost = False, strategy_type = None, use_batch_norm = None, \
        kernel_initializer = "he_uniform", \
        activation_dense = "relu", activation_output = "linear", 
        delta_constraint = None, share_strategy_across_time = False, 
        cost_structure = "proportional"):
        
    # State variables.
    prc = Input(shape=(num_instr,), name = "prc_0")
    information_set = Input(shape=(num_instr,), name = "information_set_0")

    if is_barrier:
      barrier_hit = Input(shape=(1,), name = "barrier_hit_0")
      inputs = [prc, information_set,barrier_hit]
    else:
      inputs = [prc, information_set]
    
    for j in range(N+1):            
        if j < N:
            # Define the inputs for the strategy layers here.
            if strategy_type == "simple":
                if is_barrier:
                  inputs_strategy = [information_set,barrier_hit]  
                else:
                  inputs_strategy = information_set
            elif strategy_type == "recurrent":
                if j == 0:
                    # Strategy at t = -1 should be 0. 
                    scalar = np.zeros(shape=(1,1))*1
                    strategy = Lambda(lambda x: x[0]*x[1], output_shape=lambda x:x[0])([prc,scalar])
                if is_barrier:
                    inputs_strategy = [Concatenate()([information_set,strategy]),barrier_hit] 
                else:
                    inputs_strategy = Concatenate()([information_set,strategy])


            # Determine if the strategy function depends on time t or not.
            if not share_strategy_across_time:
                strategy_layer = Strategy_Layer(d = d, m = m, num_instr = num_instr, \
                         is_barrier = is_barrier, use_batch_norm = use_batch_norm, \
                         kernel_initializer = kernel_initializer, \
                         activation_dense = activation_dense, \
                         activation_output = activation_output, 
                         delta_constraint = delta_constraint, \
                         day = j)
            else:
                if j == 0:
                    # Strategy does not depend on t so there's only a single
                    # layer at t = 0
                    strategy_layer = Strategy_Layer(d = d, m = m, num_instr = num_instr, \
                             is_barrier = is_barrier, use_batch_norm = use_batch_norm, \
                             kernel_initializer = kernel_initializer, \
                             activation_dense = activation_dense, \
                             activation_output = activation_output, 
                             delta_constraint = delta_constraint, \
                             day = j)
            
            strategyhelper = strategy_layer(inputs_strategy)
            
            # strategy_-1 is set to 0
            # delta_strategy = strategy_{t+1} - strategy_t
            if j == 0:              
                delta_strategy = strategyhelper
            else:
                delta_strategy = Subtract(name = "diff_strategy_" + str(j))([strategyhelper, strategy])
            
            if cost_structure == "proportional": 
                # Proportional transaction costs
                absolutechanges = Lambda(lambda x : Kbackend.abs(x), name = "absolutechanges_" + str(j))(delta_strategy)
                costs = Lambda(lambda x : epsilon*x, name = "costs_step1_" + str(j))(absolutechanges)
                costs = Dot(name = "costs_" + str(j),axes=1)([costs,prc])
            elif cost_structure == "constant":
                # Constant transaction costs
                costs = Lambda(lambda x : epsilon + x*0.0)(prc)
                    
            if j == 0:
                wealth = Lambda(lambda x : initial_wealth - x, name = "costDot_" + str(j))(costs)
            else:
                wealth = Subtract(name = "costDot_" + str(j))([wealth, costs])
            
            # Wealth for the next period
            # w_{t+1} = w_t + (strategy_t-strategy_{t+1})*prc_t
            #         = w_t - delta_strategy*prc_t
            mult = Dot(axes=1)([delta_strategy, prc])
            wealth = Subtract(name = "wealth_" + str(j))([wealth, mult])

            # Accumulate interest rate for next period.
            FV_factor = np.exp(risk_free*dt)
            wealth = Lambda(lambda x: x*FV_factor)(wealth)
            
            prc = Input(shape=(num_instr,),name = "prc_" + str(j+1))
            information_set = Input(shape=(num_instr,), name = "information_set_" + str(j+1))

            if is_barrier:
                barrier_hit = Input(shape=(1,), name = "barrier_hit_" + str(j+1))

            strategy = strategyhelper    
            
            if j != N - 1:
                if is_barrier:
                    inputs += [prc, information_set,barrier_hit]
                else: 
                    inputs += [prc, information_set]
            else:
                inputs += [prc]
        else:
            # The paper assumes no transaction costs for the final period 
            # when the position is liquidated.
            if final_period_cost:
                if cost_structure == "proportional":
                    # Proportional transaction costs.
                    absolutechanges = Lambda(lambda x : Kbackend.abs(x), name = "absolutechanges_" + str(j))(strategy)
                    costs = Lambda(lambda x : epsilon*x, name = "costs_step1_" + str(j))(absolutechanges)
                    costs = Dot(name = "costs_" + str(j),axes=1)([costs,prc])
                elif cost_structure == "constant":
                    # Constant transaction costs.
                    costs = Lambda(lambda x : epsilon + x*0.0)(prc)

                wealth = Subtract(name = "costDot_" + str(j))([wealth, costs])
            # Wealth for the final period
            # -delta_strategy = strategy_t
            mult = Dot(axes=1)([strategy, prc])
            wealth = Add()([wealth, mult])
                 
            # Add the terminal payoff of any derivatives.
            payoff = Input(shape=(1,), name = "payoff")
            inputs += [payoff]
            
            wealth = Add(name = "wealth_" + str(j))([wealth,payoff])
    return Model(inputs=inputs, outputs=wealth)

def Delta_SubModel(model = None, days_from_today = None, share_strategy_across_time = False):
    inputs = model.get_layer("delta_" + str(days_from_today)).input
        
    if not share_strategy_across_time:
        outputs = model.get_layer("delta_" + str(days_from_today))(inputs)
    else:
        outputs = model.get_layer("delta_0")(inputs)
        
    return Model(inputs, outputs)
