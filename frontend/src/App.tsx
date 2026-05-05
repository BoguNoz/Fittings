import {compositeStore} from "./stores/composite-store.ts";
import {appComposites} from "./repositories/composites.ts";

const App = () => {

  compositeStore.initializeComposite(appComposites)

  return (
      <p>hello</p>
  )
}

export default App
