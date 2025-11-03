import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Chapter 3, Section 2: Data Simulation
# -------------------------------------------------------------------
def generate_synthetic_data(num_users=1000):
    """
    Generate two synthetic dataframes for Hamrah-e Avval (telecom) and Bank
    """
    np.random.seed(42)
    
    # 1. Common user identifiers
    user_ids = range(1, num_users + 1)
    
    # 2. Hamrah-e Avval data (Organization A - Guest)
    data_usage_gb = np.random.normal(15, 5, num_users).clip(1, 50)
    charge_frequency = np.random.randint(1, 5, num_users)
    hamrah_data = pd.DataFrame({
        'user_id': user_ids,
        'data_usage_gb': data_usage_gb,
        'charge_frequency': charge_frequency
    })
    
    # 3. Bank data (Organization B - Host)
    account_balance = np.random.normal(500, 200, num_users).clip(0, 2000)
    loan_history_score = np.random.rand(num_users) # 0 (bad) to 1 (good)
    
    # Create labels (loan eligibility) based on a combination of features from both organizations
    # This hidden logic is what the model needs to learn
    combined_score = (0.2 * data_usage_gb / 50 +
                      0.1 * charge_frequency / 5 +
                      0.4 * account_balance / 2000 +
                      0.3 * loan_history_score)
    
    # Convert score to probability and then to binary label
    probability = 1 / (1 + np.exp(-( (combined_score - 0.5) * 10 ))) # Sigmoid function
    is_eligible_for_loan = (probability > 0.5).astype(int)
    
    bank_data = pd.DataFrame({
        'user_id': user_ids,
        'account_balance': account_balance,
        'loan_history_score': loan_history_score,
        'is_eligible_for_loan': is_eligible_for_loan
    })
    
    print(f"Data generation complete. {num_users} common users.")
    print("\n--- Hamrah-e Avval Data (Sample) ---")
    print(hamrah_data.head())
    print("\n--- Bank Data (Sample) ---")
    print(bank_data.head())
    
    return hamrah_data, bank_data

# -------------------------------------------------------------------
# Chapter 3, Section 4: Blockchain Interaction Simulation
# -------------------------------------------------------------------
class BlockchainSimulator:
    """
    This class simulates logging events to a blockchain.
    In Phase 3, this class will be replaced with real smart contract calls.
    """
    def __init__(self):
        self.ledger = []
        self.block_height = 0
        print("\n[BlockchainSim] Consortium Blockchain network is active.")
        
    def log_event(self, event_type, details):
        """
        Log an event as a transaction in the ledger
        """
        self.block_height += 1
        log_entry = {
            'block': self.block_height,
            'timestamp': pd.Timestamp.now(),
            'event_type': event_type,
            'details': details
        }
        self.ledger.append(log_entry)
        print(f"[BlockchainSim] Event '{event_type}' logged in Block {self.block_height}.")
        
    def print_ledger(self):
        print("\n--- Blockchain Ledger (Audit Trail) ---")
        for entry in self.ledger:
            print(f"  Block {entry['block']}: {entry['event_type']} - {entry['details']}")
        print("-----------------------------------------")

# -------------------------------------------------------------------
# Chapter 3, Section 3: VFL Process Simulation (Simple Logistic Regression)
# -------------------------------------------------------------------

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

class VFLGuest:
    """Organization A: Hamrah-e Avval (Guest)"""
    def __init__(self, data, learning_rate=0.01):
        self.X = self._preprocess(data.drop(columns=['user_id']))
        self.weights = np.random.rand(self.X.shape[1], 1) * 0.01 # W_a
        self.lr = learning_rate
        self.public_key = "GUEST_PUB_KEY" # In real world, used for encryption
        print(f"[Guest: Hamrah] Initialized. Features: {self.X.shape[1]}, Weights shape: {self.weights.shape}")
        
    def _preprocess(self, data):
        # Simple normalization
        data = (data - data.mean()) / data.std()
        # Add bias term
        data['bias'] = 1
        return data.values
        
    def compute_partial_z(self):
        """Compute local partial z (z_a = X_a * W_a)"""
        self.z_a = self.X.dot(self.weights)
        # In real world, z_a would be encrypted before sending
        print("[Guest: Hamrah] Computed partial z_a.")
        return self.z_a # Send to coordinator

    def update_weights(self, guest_gradient):
        """Update weights based on gradient received from host"""
        # Received gradient should be decrypted
        update = self.lr * guest_gradient
        self.weights -= update
        # print("[Guest: Hamrah] Weights (W_a) updated.")


class VFLHost:
    """Organization B: Bank (Host)"""
    def __init__(self, data, learning_rate=0.01):
        self.X = self._preprocess(data.drop(columns=['user_id', 'is_eligible_for_loan']))
        self.y = data['is_eligible_for_loan'].values.reshape(-1, 1)
        self.weights = np.random.rand(self.X.shape[1], 1) * 0.01 # W_b
        self.lr = learning_rate
        self.public_key = "HOST_PUB_KEY" # For encryption
        print(f"[Host: Bank] Initialized. Features: {self.X.shape[1]}, Weights shape: {self.weights.shape}")
        
    def _preprocess(self, data):
        data = (data - data.mean()) / data.std()
        data['bias'] = 1
        return data.values

    def compute_loss_and_gradients(self, guest_z_a):
        """
        Compute complete z, loss, and gradients after receiving partial z from guest
        """
        # 1. Receive z_a (should be decrypted) and compute z_b
        z_b = self.X.dot(self.weights)
        
        # 2. Aggregate z (this should be done securely by the coordinator)
        z_total = guest_z_a + z_b
        
        # 3. Compute prediction and loss
        y_hat = sigmoid(z_total)
        loss = -np.mean(self.y * np.log(y_hat + 1e-9) + (1 - self.y) * np.log(1 - y_hat + 1e-9))
        
        # 4. Compute common gradient
        common_gradient = (y_hat - self.y) / self.y.shape[0]
        
        # 5. Split gradients
        # Gradient for host (bank)
        host_gradient = self.X.T.dot(common_gradient) # dL/dW_b
        
        # Gradient for guest (Hamrah-e Avval) - this should be securely returned
        # In this simple simulation, we don't have X_a, but we know dL/dz_a = common_gradient
        # and dL/dW_a = X_a.T * dL/dz_a
        # We return dL/dz_a (i.e., common_gradient) so the guest can compute dL/dW_a itself
        # (In real implementation, this is different and X_a.T * common_gradient is computed)
        # For simplicity, we assume the guest can compute its weight gradient with common_gradient
        
        # Actually, the guest should compute X_a.T.dot(common_gradient)
        # The coordinator sends common_gradient to the guest
        
        print(f"[Host: Bank] Loss calculated: {loss:.4f}")
        
        # 6. Update host weights
        update = self.lr * host_gradient
        self.weights -= update
        # print("[Host: Bank] Weights (W_b) updated.")
        
        # 7. Return gradient for guest (should be encrypted)
        return loss, common_gradient 

# -------------------------------------------------------------------
# Chapter 3: Main Training Loop
# -------------------------------------------------------------------
def main_training_loop(epochs=20):
    print("--- Phase 2: VFL Proof-of-Concept Simulation ---")
    
    # 1. Initialize blockchain
    blockchain = BlockchainSimulator()
    blockchain.log_event("SYSTEM_INIT", {"participants": ["Hamrah-e Avval", "Bank"]})
    
    # 2. Generate data
    hamrah_data, bank_data = generate_synthetic_data(num_users=1000)
    blockchain.log_event("DATA_PREPARATION", {"user_count": len(hamrah_data)})
    
    # 3. Initialize VFL parties
    guest = VFLGuest(hamrah_data, learning_rate=0.1)
    host = VFLHost(bank_data, learning_rate=0.1)
    
    # 4. Start training process
    job_id = "loan_model_v1"
    blockchain.log_event("TRAINING_START", {"job_id": job_id, "epochs": epochs, "model_type": "VFL_LogisticRegression"})

    # --- Training loop ---
    for i in range(epochs):
        print(f"\n--- Epoch {i+1}/{epochs} ---")
        
        # Simulate user alignment (here we use all data)
        
        # 1. Guest computes and sends z_a
        z_a = guest.compute_partial_z()
        # (Secure communication: z_a is encrypted and sent to coordinator)
        
        # 2. Host computes z_b, loss and gradients
        # (Coordinator aggregates z_a and z_b and gives to host)
        loss, common_gradient = host.compute_loss_and_gradients(z_a)
        
        # 3. Host returns gradient for guest
        # (Secure communication: common_gradient is encrypted and sent to coordinator then to guest)
        
        # 4. Guest computes its gradient and updates weights
        # (Guest receives and decrypts common_gradient)
        guest_gradient = guest.X.T.dot(common_gradient) # dL/dW_a
        guest.update_weights(guest_gradient)
        
        # 5. Log events to blockchain
        blockchain.log_event("EPOCH_COMPLETE", {"job_id": job_id, "epoch": i+1, "loss": f"{loss:.6f}"})

    # --- Training end ---
    final_model_hash = f"W_a: {hash(guest.weights.tobytes())} | W_b: {hash(host.weights.tobytes())}"
    blockchain.log_event("TRAINING_END", {"job_id": job_id, "final_loss": f"{loss:.6f}", "model_hash": final_model_hash})
    
    # Display final results
    print("\n--- Training Finished ---")
    print(f"Final Loss: {loss:.4f}")
    blockchain.print_ledger()

if __name__ == "__main__":
    main_training_loop(epochs=15)
