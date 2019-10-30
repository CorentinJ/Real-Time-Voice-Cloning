import React, {Component} from 'react';
import logo from './logo.svg';
import axios from 'axios';
import './App.css';
import * as audio from './audio-file.flac';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      fileuploaded: null,
      confidence: null,
      transcript: null
    }
  }

  handlefile(e){
    this.setState({fileuploaded: e.target.files[0]});
  }


  async getDataAxios(){
    var files = this.state.fileuploaded;
    var c, t;
    await axios({
      method: 'POST',
      url: 'https://stream.watsonplatform.net/speech-to-text/api/v1/recognize',
      qs: { 'Content-Type': 'audio/flac' },
      auth: {
        username: 'apikey',
        password: 'U9nQSXMUvHLonqi2CBoqAl-Pcbz9tbCXuK3rWVYcPaIY'
      },
      data: files,
      headers: { 'Content-Type': 'audio/flac' },
    }).then(function (response) {
      // handle success
      c = response.data.results[0].alternatives[0].confidence;
      t = response.data.results[0].alternatives[0].transcript;
    })
    .catch(function (error) {
      // handle error
      console.log(error);
    })

    this.setState({confidence: c, transcript: t});
  }
  render(){
    return (
      <div className="App">
        <header className="App-header">

          <input type="file" name="file" onChange={(e) => this.handlefile(e)} />
          <button type="submit" onClick={()=>this.getDataAxios()} >Add</button>
          <br/>
          Confidence:<h3> {this.state.confidence}</h3>
          Transcript:<h3> {this.state.transcript}</h3>
        </header>
      </div>
    );
  }
}

export default App;
