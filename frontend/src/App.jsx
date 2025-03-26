import {useState} from 'react';
import {BrowserRouter, Routes, Route, Navigate} from 'react-router-dom';
import Login from './components/Login';
import Capture from './components/Capture'

function App() {
    const [user, setUser] = useState(null);

    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={user ? <Navigate to="/capture"/> : <Login setUser={setUser} />}/>
                <Route path="/capture" element={user ? <Capture user={user.identifier} userHeight={user.height} onLogout={() => setUser(null)}/> : <Navigate to="/"/>}/>
            </Routes>
        </BrowserRouter>
    )
}

export default App;