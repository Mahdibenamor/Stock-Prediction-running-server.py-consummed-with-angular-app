import {Component, OnInit} from '@angular/core';
import { FormGroup,FormBuilder,Validators } from '@angular/forms';
import {HttpClient} from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit{
  price:any="";
  title = 'deploy-app-ng';
  userForm :FormGroup;
  constructor(private formBuilder: FormBuilder,
              public  http:HttpClient
  ){}




  ngOnInit() {
    this.initForm();
  }

  initForm() {
    this.userForm = this.formBuilder.group({
      inputs: []
    });
  }
  submit(){
    const St = this.userForm.get('inputs').value;
    let toArray=St.substring(1 ,St.length-1 ).split(",");
    var inputs = toArray.map(num => Number(num));
    this.http.post('http://localhost:5000/results',{"inputs":inputs}).subscribe(
      (data)=>{console.log(data)
                        this.price=data},
      (err)=>{console.log(err)},
      ()=>{} )

  }
}

